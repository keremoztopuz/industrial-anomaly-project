import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile, Request
import uvicorn

from mvtec_dataset import build_transform
from patchcore import PatchCore

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
transform = build_transform(256)  # Example image size, adjust as needed

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
async def predict(category: str, upload_file: UploadFile):
    contents = await upload_file.read()
    return {
        "category": category,
        "filename": upload_file.filename,
        "size_bytes": len(contents),
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
