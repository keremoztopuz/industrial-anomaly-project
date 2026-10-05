"""Score one image and write the prediction log line."""

import json
import time
from functools import lru_cache

import torch
from PIL import ImageStat

from anomaly.data.mvtec_dataset import build_transform
from api.config import settings


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
    print(
        json.dumps(
            {
                "event": "prediction",
                "category": category,
                "model_version": settings.model_version,
                "anomaly_score": round(score, 2),
                "threshold": threshold,
                "is_anomaly": is_anomaly,
                "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                "width": image.width,
                "height": image.height,
                "brightness": round(stats.mean[0], 1),
                "contrast": round(stats.stddev[0], 1),
            }
        ),
        flush=True,
    )
    return {
        "category": category,
        "anomaly_score": score,
        "threshold": threshold,
        "is_anomaly": is_anomaly,
    }
