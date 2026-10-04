import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile, Request, HTTPException
import uvicorn

from mvtec_dataset import build_transform
from patchcore import PatchCore

from io import BytesIO
from PIL import Image

import torch

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "artifacts/border-exclusion/coreset-16384-n3-b2/patchcore"))
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", 8000))

def load_thresholds(path):
    """Read per-category thresholds written by scripts/calibrate_thresholds.py, if present."""
    if not path.is_file():
        print(f"No thresholds at {path}; is_anomaly will be null")
        return {}
    categories = json.loads(path.read_text(encoding="utf-8"))["categories"]
    return {category: values["threshold"] for category, values in categories.items()}

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

    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        scores, _ = models[category].predict(tensor)

    score = scores[0].item()
    threshold = request.app.state.thresholds.get(category)
    return {
        "category": category,
        "filename": upload_file.filename,
        "anomaly_score": score,
        "threshold": threshold,
        "is_anomaly": None if threshold is None else score > threshold,
    }

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
