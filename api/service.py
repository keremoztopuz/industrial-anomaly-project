from fastapi import FastAPI, UploadFile
from mvtec_dataset import build_transform
import uvicorn

app = FastAPI()
transform = build_transform(256)  # Example image size, adjust as needed

@app.get("/")
def read_root():
    return {"status": "API is running"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/categories")
def get_categories():
    return {"categories": ["category1", "category2", "category3"]}

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
