"""A fake model, a client without model loading and in-memory images for API
tests."""

from io import BytesIO

import torch
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app


class FakeModel:
    """Stand-in for PatchCore: a fixed score, remembering the batch it got."""

    def __init__(self, score, image_size=256):
        self.score = score
        self.image_size = image_size
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
