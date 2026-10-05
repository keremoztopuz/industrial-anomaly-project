"""HTTP parameters, response documentation and error mapping."""

from fastapi import APIRouter, HTTPException, Request, UploadFile

from api import config, services
from api.schemas import (
    CategoriesResponse, ErrorResponse, ModelResponse, PredictionResponse, StatusResponse,
    ValidationErrorResponse,
)

router = APIRouter()


@router.get("/", response_model=StatusResponse, summary="API status")
def read_root():
    """Report that the HTTP API is running."""
    return {"status": "API is running"}


@router.get("/health", response_model=StatusResponse, summary="Check HTTP liveness")
def check_liveness():
    """Report process liveness; models are validated during startup."""
    return {"status": "healthy"}


@router.get("/categories", response_model=CategoriesResponse,
            summary="List loaded product categories")
def list_loaded_product_categories(request: Request):
    """Return the categories available for prediction, for example bottle and cable."""
    return {"categories": sorted(request.app.state.models)}


@router.get("/model", response_model=ModelResponse, summary="Get deployed model information")
def get_deployed_model_info(request: Request):
    """Return the model identity, storage directory and number of loaded categories."""
    return {"name": config.MODEL_NAME, "version": config.MODEL_VERSION,
            "model_dir": str(config.MODEL_DIR), "categories": len(request.app.state.models)}


@router.post("/predict/{category}", response_model=PredictionResponse,
             summary="Predict an image anomaly", responses={
                 400: {"model": ErrorResponse, "description": "Invalid or unsupported image"},
                 404: {"model": ErrorResponse, "description": "Category model is not loaded"},
                 413: {"model": ErrorResponse, "description": "Byte or pixel limit exceeded"},
                 422: {"model": ValidationErrorResponse,
                       "description": "Missing upload or invalid request parameters"},
             })
def predict_image_anomaly(category: str, upload_file: UploadFile, request: Request):
    """Score a PNG/JPEG image; return a null decision if calibration is absent."""
    models = request.app.state.models
    if category not in models:
        raise HTTPException(404, f"Category '{category}' not found")
    try:
        image = services.decode_image(upload_file.file)
    except services.ImageTooLarge as error:
        raise HTTPException(413, str(error)) from error
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    with image:
        result = services.predict_image(
            models[category], image, category, request.app.state.thresholds.get(category))
    return {**result, "filename": upload_file.filename}
