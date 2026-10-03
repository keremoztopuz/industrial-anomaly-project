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

    return {
        "category": category,
        "filename": upload_file.filename,
        "anomaly_score": scores[0].item(),
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
