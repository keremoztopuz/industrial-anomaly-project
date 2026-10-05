"""Model loading, bounded image decoding and prediction logging."""

import json
import math
import time
from functools import lru_cache
from io import BytesIO

import torch
from PIL import Image, ImageStat

from anomaly.mvtec_dataset import build_transform
from anomaly.patchcore import PatchCore
from api import config


class ImageTooLarge(ValueError):
    """The encoded file or decoded dimensions exceed the configured limit."""


def load_thresholds(path):
    """Allow missing calibration; reject malformed or non-finite thresholds."""
    if not path.exists():
        print(f"No thresholds at {path}; is_anomaly will be null")
        return {}
    try:
        categories = json.loads(path.read_text(encoding="utf-8"))["categories"]
        if not isinstance(categories, dict):
            raise ValueError("categories must be an object")
        thresholds = {}
        for category, values in categories.items():
            value = values["threshold"]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{category}: threshold must be a finite nonnegative number")
            thresholds[category] = value
        return thresholds
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(f"Invalid thresholds at {path}: {error}") from error


def load_models(directory):
    models, backbone = {}, None
    for path in sorted(directory.glob("*.pt")):
        model = PatchCore.load(path, device="cpu", backbone=backbone)
        backbone = model.backbone
        models[path.stem] = model
        print(f"Loaded model {path.stem} from {path}")
    if not models:
        raise RuntimeError(f"No category models (*.pt) found in MODEL_DIR={directory}")
    return models


def decode_image(file):
    contents = file.read(config.MAX_UPLOAD_BYTES + 1)
    if len(contents) > config.MAX_UPLOAD_BYTES:
        raise ImageTooLarge("Image file exceeds the byte limit")
    try:
        with Image.open(BytesIO(contents), formats=("PNG", "JPEG")) as image:
            if image.width * image.height > config.MAX_IMAGE_PIXELS:
                raise ImageTooLarge("Image exceeds the pixel limit")
            return image.convert("RGB")
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise ImageTooLarge("Image exceeds the pixel limit") from error
    except (OSError, SyntaxError) as error:
        raise ValueError("Invalid image file") from error


@lru_cache
def transform_for(image_size):
    """Cache preprocessing by the size each bank was trained with."""
    return build_transform(image_size)


def predict_image(model, image, category, threshold):
    stats = ImageStat.Stat(image.convert("L"))
    start = time.perf_counter()
    tensor = transform_for(model.image_size)(image).unsqueeze(0)
    with torch.no_grad():
        scores, _ = model.predict(tensor)
    score = scores[0].item()
    is_anomaly = None if threshold is None else score > threshold
    print(json.dumps({
        "event": "prediction", "category": category, "model_version": config.MODEL_VERSION,
        "anomaly_score": round(score, 2), "threshold": threshold, "is_anomaly": is_anomaly,
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        "width": image.width, "height": image.height,
        "brightness": round(stats.mean[0], 1), "contrast": round(stats.stddev[0], 1),
    }), flush=True)
    return {"category": category, "anomaly_score": score,
            "threshold": threshold, "is_anomaly": is_anomaly}
