from fastapi import FastAPI, UploadFile
import uvicorn

app = FastAPI()

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
    uvicorn.run(app, host="0.0.0.0", port=8000)
