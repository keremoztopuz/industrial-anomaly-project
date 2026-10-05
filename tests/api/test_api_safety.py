import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient
from PIL import Image

from api import config
from api.main import app
from api.middleware import RequestSizeLimit
from api.services import image_decoding, model_loading
from tests.support.api_fakes import FakeModel, image_bytes, make_client


class ImageSafetyTests(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel(0.5)
        self.client = make_client({"bottle": self.model})

    def upload(self, contents):
        return self.client.post("/predict/bottle", files={"upload_file": ("image.png", contents)})

    def test_limits_prevent_prediction(self):
        for setting, limit in (("MAX_UPLOAD_BYTES", 10), ("MAX_IMAGE_PIXELS", 10),
                               ("MAX_REQUEST_BYTES", 10)):
            with self.subTest(setting=setting), mock.patch.object(config, setting, limit):
                self.assertEqual(self.upload(image_bytes()).status_code, 413)
                self.assertIsNone(self.model.last_batch)

    def test_exact_byte_and_pixel_limits_are_allowed(self):
        contents = image_bytes()
        with mock.patch.object(config, "MAX_UPLOAD_BYTES", len(contents)), \
                mock.patch.object(config, "MAX_IMAGE_PIXELS", 64 * 48):
            self.assertEqual(self.upload(contents).status_code, 200)

    def test_pillow_decompression_bomb_is_413(self):
        with mock.patch.object(Image, "MAX_IMAGE_PIXELS", 10):
            self.assertEqual(self.upload(image_bytes()).status_code, 413)
        self.assertIsNone(self.model.last_batch)

    def test_unsupported_format_is_400_even_with_png_filename(self):
        self.assertEqual(self.upload(image_bytes(fmt="GIF")).status_code, 400)
        self.assertIsNone(self.model.last_batch)

    def test_bounded_read(self):
        file = mock.Mock()
        file.read.return_value = b"x" * 11
        with mock.patch.object(config, "MAX_UPLOAD_BYTES", 10), \
                self.assertRaises(image_decoding.ImageTooLarge):
            image_decoding.decode_image(file)
        file.read.assert_called_once_with(11)

    def test_chunked_and_false_content_length_checked_before_parser(self):
        async def run(headers):
            downstream = mock.AsyncMock()
            middleware = RequestSizeLimit(downstream)
            receive = mock.AsyncMock(side_effect=[
                {"type": "http.request", "body": b"123456", "more_body": True},
                {"type": "http.request", "body": b"78901", "more_body": False},
            ])
            send = mock.AsyncMock()
            with mock.patch.object(config, "MAX_REQUEST_BYTES", 10):
                await middleware({"type": "http", "headers": headers}, receive, send)
            downstream.assert_not_awaited()
            self.assertEqual(send.call_args_list[0].args[0]["status"], 413)

        for headers in ([], [(b"content-length", b"1")], [(b"content-length", b"11")]):
            asyncio.run(run(headers))

    def test_openapi_has_response_contracts_and_errors(self):
        schema = app.openapi()
        route = schema["paths"]["/predict/{category}"]["post"]
        for code in ("200", "400", "404", "413", "422"):
            self.assertIn(code, route["responses"])
            self.assertIn("schema", route["responses"][code]["content"]["application/json"])
        fields = schema["components"]["schemas"]["PredictionResponse"]["properties"]
        for field in ("threshold", "is_anomaly"):
            self.assertIn({"type": "null"}, fields[field]["anyOf"])
        self.assertIn("description", route)


class StartupTests(unittest.TestCase):
    def test_empty_directory_fails_real_lifespan(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(config, "MODEL_DIR", Path(directory)), \
                self.assertRaisesRegex(RuntimeError, "No category models"):
            with TestClient(app):
                pass

    def test_startup_loads_models_and_missing_calibration(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(config, "MODEL_DIR", Path(directory)), \
                mock.patch.object(model_loading.PatchCore, "load") as load:
            (Path(directory) / "bottle.pt").touch()
            load.return_value = FakeModel(0.5)
            load.return_value.backbone = object()
            with TestClient(app) as client:
                self.assertEqual(client.get("/categories").json(), {"categories": ["bottle"]})
                response = client.post("/predict/bottle", files={
                    "upload_file": ("image.png", image_bytes())})
                self.assertIsNone(response.json()["is_anomaly"])
            load.assert_called_once()

    def test_invalid_thresholds_fail_startup(self):
        malformed = ["invalid json", "[]", '{}', '{"categories": []}']
        malformed += [json.dumps({"categories": {"bottle": {"threshold": value}}})
                      for value in (None, True, "1", -1, float("nan"), float("inf"))]
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(config, "MODEL_DIR", Path(directory)), \
                mock.patch.object(model_loading, "load_models",
                                  return_value={"bottle": FakeModel(1)}):
            path = Path(directory) / "thresholds.json"
            for text in malformed:
                path.write_text(text)
                with self.subTest(text=text), self.assertRaisesRegex(
                        ValueError, "Invalid thresholds"):
                    with TestClient(app):
                        pass
