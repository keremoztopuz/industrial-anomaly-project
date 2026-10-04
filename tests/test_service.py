import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import BytesIO, StringIO
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "api"))

from fastapi.testclient import TestClient

from service import app, load_thresholds


class FakeModel:
    """Stand-in for PatchCore: returns a fixed score and remembers its input."""

    def __init__(self, score):
        self.score = score
        self.last_batch = None

    def predict(self, images):
        self.last_batch = images
        batch_size, _, height, width = images.shape
        scores = torch.full((batch_size,), self.score)
        maps = torch.zeros((batch_size, height, width))
        return scores, maps


def make_client(models, thresholds=None):
    """Return a TestClient serving `models` without loading real .pt files.

    TestClient only runs the lifespan inside a `with` block, so we skip it
    and set app.state.models and app.state.thresholds ourselves.
    """
    app.state.models = models
    app.state.thresholds = thresholds or {}
    return TestClient(app)


def image_bytes(mode="RGB", size=(64, 48), fmt="PNG"):
    """Encode a blank in-memory image so tests need no dataset files."""
    buffer = BytesIO()
    Image.new(mode, size).save(buffer, format=fmt)
    return buffer.getvalue()


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.bottle = FakeModel(score=0.5)
        self.client = make_client({"bottle": self.bottle, "cable": FakeModel(score=1.0)},
                                  thresholds={"bottle": 0.4, "cable": 2.0})

    def predict(self, category, contents, filename="image.png"):
        return self.client.post(f"/predict/{category}",
                                files={"upload_file": (filename, contents)})

    def test_root_and_health(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/health").json(), {"status": "healthy"})

    def test_model_reports_name_version_dir_and_loaded_categories(self):
        body = self.client.get("/model").json()
        self.assertEqual(body["name"], "patchcore-mvtec")
        self.assertEqual(body["version"], "local")
        self.assertEqual(body["categories"], 2)
        self.assertIn("model_dir", body)

    def test_categories_are_sorted(self):
        response = self.client.get("/categories")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"categories": ["bottle", "cable"]})

    def test_predict_returns_score(self):
        response = self.predict("bottle", image_bytes())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "category": "bottle",
            "filename": "image.png",
            "anomaly_score": 0.5,
            "threshold": 0.4,
            "is_anomaly": True,
        })

    def test_score_at_or_below_threshold_is_normal(self):
        self.assertFalse(self.predict("cable", image_bytes()).json()["is_anomaly"])
        client = make_client({"bottle": FakeModel(score=0.5)}, thresholds={"bottle": 0.5})
        response = client.post("/predict/bottle", files={"upload_file": ("a.png", image_bytes())})
        self.assertFalse(response.json()["is_anomaly"])

    def test_missing_threshold_returns_null_decision(self):
        client = make_client({"bottle": FakeModel(score=0.5)})
        body = client.post("/predict/bottle",
                           files={"upload_file": ("a.png", image_bytes())}).json()
        self.assertEqual(body["anomaly_score"], 0.5)
        self.assertIsNone(body["threshold"])
        self.assertIsNone(body["is_anomaly"])

    def test_predict_uses_requested_category(self):
        response = self.predict("cable", image_bytes())
        self.assertEqual(response.json()["anomaly_score"], 1.0)
        self.assertIsNone(self.bottle.last_batch)

    def test_predict_sends_normalized_batch(self):
        self.predict("bottle", image_bytes(size=(300, 200)))
        self.assertEqual(tuple(self.bottle.last_batch.shape), (1, 3, 256, 256))
        self.assertEqual(self.bottle.last_batch.dtype, torch.float32)

    def test_predict_converts_grayscale_and_rgba_to_rgb(self):
        for mode in ("L", "RGBA"):
            with self.subTest(mode=mode):
                response = self.predict("bottle", image_bytes(mode=mode))
                self.assertEqual(response.status_code, 200)
                self.assertEqual(self.bottle.last_batch.shape[1], 3)

    def test_predict_accepts_jpeg(self):
        response = self.predict("bottle", image_bytes(fmt="JPEG"), filename="image.jpg")
        self.assertEqual(response.status_code, 200)

    def logged(self, category, contents, filename="image.png"):
        output = StringIO()
        with redirect_stdout(output):
            response = self.predict(category, contents, filename)
        lines = [json.loads(line) for line in output.getvalue().splitlines()
                 if line.startswith("{")]
        return response, [line for line in lines if line.get("event") == "prediction"]

    def test_prediction_is_logged_as_one_json_line(self):
        response, [line] = self.logged("bottle", image_bytes(mode="L", size=(64, 48)),
                                       filename="secret-order-42.png")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(line["category"], "bottle")
        self.assertEqual(line["model_version"], "local")
        self.assertEqual(line["anomaly_score"], 0.5)
        self.assertEqual(line["threshold"], 0.4)
        self.assertTrue(line["is_anomaly"])
        self.assertEqual((line["width"], line["height"]), (64, 48))
        self.assertEqual((line["brightness"], line["contrast"]), (0.0, 0.0))
        self.assertGreaterEqual(line["latency_ms"], 0)
        self.assertNotIn("filename", line)
        self.assertNotIn("secret-order-42", json.dumps(line))

    def test_log_and_response_agree_on_is_anomaly(self):
        response, [line] = self.logged("cable", image_bytes())
        self.assertEqual(line["is_anomaly"], response.json()["is_anomaly"])
        self.assertFalse(line["is_anomaly"])

    def test_rejected_requests_are_not_logged_as_predictions(self):
        self.assertEqual(self.logged("banana", image_bytes())[1], [])
        self.assertEqual(self.logged("bottle", b"not an image", "notes.txt")[1], [])

    def test_unknown_category_returns_404(self):
        response = self.predict("banana", image_bytes())
        self.assertEqual(response.status_code, 404)
        self.assertIsNone(self.bottle.last_batch)

    def test_non_image_returns_400(self):
        response = self.predict("bottle", b"not an image", filename="notes.txt")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"detail": "Invalid image file"})

    def test_truncated_image_returns_400(self):
        response = self.predict("bottle", image_bytes()[:40])
        self.assertEqual(response.status_code, 400)

    def test_empty_file_returns_400(self):
        self.assertEqual(self.predict("bottle", b"").status_code, 400)

    def test_missing_file_returns_422(self):
        response = self.client.post("/predict/bottle")
        self.assertEqual(response.status_code, 422)


class LoadThresholdsTests(unittest.TestCase):
    def test_reads_threshold_per_category(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "thresholds.json"
            path.write_text(json.dumps({"method": "max", "categories": {
                "bottle": {"threshold": 12.5, "holdout_images": 42},
                "cable": {"threshold": 20.0, "holdout_images": 45},
            }}), encoding="utf-8")
            self.assertEqual(load_thresholds(path), {"bottle": 12.5, "cable": 20.0})

    def test_missing_file_gives_no_thresholds(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(load_thresholds(Path(directory) / "thresholds.json"), {})


if __name__ == "__main__":
    unittest.main()
