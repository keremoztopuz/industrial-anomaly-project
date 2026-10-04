import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile, Request, HTTPException
import uvicorn

from anomaly.mvtec_dataset import build_transform
from anomaly.patchcore import PatchCore

from io import BytesIO
from PIL import Image, ImageStat

import torch
import time

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "artifacts/border-exclusion/coreset-16384-n3-b2/patchcore"))
MODEL_NAME = os.environ.get("MODEL_NAME", "patchcore-mvtec")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "local")
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", 8000))

def load_thresholds(path):
    """Read per-category thresholds written by scripts/calibrate_thresholds.py, if present."""
    if not path.is_file():
        print(f"No thresholds at {path}; is_anomaly will be null")
        return {}
    categories = json.loads(path.read_text(encoding="utf-8"))["categories"]
    return {category: values["threshold"] for category, values in categories.items()}

def log_prediction(**fields):
    """Write one JSON line; Cloud Run sends it to Cloud Logging as jsonPayload."""
    print(json.dumps({"event": "prediction", **fields}), flush=True)

@asynccontextmanager
async def lifespan(app):
    models = {}
    backbone = None
    for path in sorted(MODEL_DIR.glob("*.pt")):
        model = PatchCore.load(path, device="cpu", backbone=backbone)
        backbone = model.backbone
        models[path.stem] = model
        print(f"Loaded model {path.stem} from {path}")
    app.state.models = models
    app.state.thresholds = load_thresholds(MODEL_DIR / "thresholds.json")
    yield

app = FastAPI(lifespan=lifespan)
transform = build_transform(256)

@app.get("/")
def read_root():
    return {"status": "API is running"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/categories")
def get_categories(request: Request):
    return {"categories": sorted(request.app.state.models.keys())}

@app.get("/model")
def get_model(request: Request):
    return {"name": MODEL_NAME,
            "version": MODEL_VERSION,
            "model_dir": str(MODEL_DIR),
            "categories": len(request.app.state.models.keys())}

@app.post("/predict/{category}")
def predict(category: str, upload_file: UploadFile, request: Request):
    models = request.app.state.models
    if category not in models:
        raise HTTPException(status_code=404, detail=f"Category '{category}' not found")

    contents = upload_file.file.read()
    try:
        image = Image.open(BytesIO(contents)).convert("RGB")
    except OSError as e:
        raise HTTPException(status_code=400, detail="Invalid image file") from e

    stats = ImageStat.Stat(image.convert("L"))

    start = time.perf_counter()
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        scores, _ = models[category].predict(tensor)

    score = scores[0].item()
    latency_ms = (time.perf_counter() - start) * 1000
    threshold = request.app.state.thresholds.get(category)
    is_anomaly = None if threshold is None else score > threshold
    log_prediction(category=category,
                   model_version=MODEL_VERSION,
                   anomaly_score=round(score, 2),
                   threshold=threshold,
                   is_anomaly=is_anomaly,
                   latency_ms=round(latency_ms, 2),
                   width=image.width,
                   height=image.height,
                   brightness=round(stats.mean[0], 1),
                   contrast=round(stats.stddev[0], 1)
                   )
    return {
        "category": category,
        "filename": upload_file.filename,
        "anomaly_score": score,
        "threshold": threshold,
        "is_anomaly": is_anomaly,
    }

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
