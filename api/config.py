"""Environment settings; .env files are not loaded automatically."""

import os
from pathlib import Path

MODEL_DIR = Path(os.environ.get(
    "MODEL_DIR", "artifacts/border-exclusion/coreset-16384-n3-b2/patchcore"))
MODEL_NAME = os.environ.get("MODEL_NAME", "patchcore-mvtec")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "local")
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 10 * 1024 * 1024))
MAX_IMAGE_PIXELS = int(os.environ.get("MAX_IMAGE_PIXELS", 16_000_000))
# Allow multipart headers in addition to the image bytes.
MAX_REQUEST_BYTES = int(os.environ.get("MAX_REQUEST_BYTES", MAX_UPLOAD_BYTES + 1024 * 1024))
if min(MAX_UPLOAD_BYTES, MAX_IMAGE_PIXELS, MAX_REQUEST_BYTES) <= 0:
    raise ValueError("Upload, pixel and request limits must be positive")
