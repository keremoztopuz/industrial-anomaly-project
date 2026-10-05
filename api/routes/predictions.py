"""The prediction endpoint: HTTP parameters and error mapping only."""

from fastapi import APIRouter, HTTPException, Request, UploadFile

from api.schemas import (
    ErrorResponse,
    PredictionResponse,
    ValidationErrorResponse,
)
from api.services import image_decoding, prediction

router = APIRouter()


@router.post(
    "/predict/{category}",
    response_model=PredictionResponse,
    summary="Predict an image anomaly",
    responses={
        400: {
            "model": ErrorResponse,
            "description": "Invalid or unsupported image",
        },
        404: {
            "model": ErrorResponse,
            "description": "Category model is not loaded",
        },
        413: {
            "model": ErrorResponse,
            "description": "Byte or pixel limit exceeded",
        },
        422: {
            "model": ValidationErrorResponse,
            "description": "Missing upload or invalid request parameters",
        },
    },
)
def predict_image_anomaly(
    category: str, upload_file: UploadFile, request: Request
):
    """Score a PNG/JPEG image; the decision is null without calibration."""
    models = request.app.state.models
    if category not in models:
        raise HTTPException(404, f"Category '{category}' not found")
    try:
        image = image_decoding.decode_image(upload_file.file)
    except image_decoding.ImageTooLarge as error:
        raise HTTPException(413, str(error)) from error
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    with image:
        result = prediction.predict_image(
            models[category],
            image,
            category,
            request.app.state.thresholds.get(category),
        )
    return {**result, "filename": upload_file.filename}
