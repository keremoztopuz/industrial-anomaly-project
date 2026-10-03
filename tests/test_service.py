import sys
import unittest
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "api"))

from fastapi.testclient import TestClient

from service import app


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


def make_client(models):
    """Return a TestClient serving `models` without loading real .pt files.

    TestClient only runs the lifespan inside a `with` block, so we skip it
    and set app.state.models ourselves.
    """
    app.state.models = models
    return TestClient(app)


def image_bytes(mode="RGB", size=(64, 48), fmt="PNG"):
    """Encode a blank in-memory image so tests need no dataset files."""
    buffer = BytesIO()
    Image.new(mode, size).save(buffer, format=fmt)
    return buffer.getvalue()


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.bottle = FakeModel(score=0.5)
        self.client = make_client({"bottle": self.bottle, "cable": FakeModel(score=1.0)})

    def predict(self, category, contents, filename="image.png"):
        return self.client.post(f"/predict/{category}",
                                files={"upload_file": (filename, contents)})

    def test_root_and_health(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/health").json(), {"status": "healthy"})

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
        })

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


if __name__ == "__main__":
    unittest.main()
