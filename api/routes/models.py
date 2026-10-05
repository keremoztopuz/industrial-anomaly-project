"""Endpoints that describe the loaded models."""

from fastapi import APIRouter, Request

from api.config import settings
from api.schemas import CategoriesResponse, ModelResponse

router = APIRouter()


@router.get(
    "/categories",
    response_model=CategoriesResponse,
    summary="List loaded product categories",
)
def list_loaded_product_categories(request: Request):
    """Return the categories available for prediction, for example bottle and
    cable."""
    return {"categories": sorted(request.app.state.models)}


@router.get(
    "/model",
    response_model=ModelResponse,
    summary="Get deployed model information",
)
def get_deployed_model_info(request: Request):
    """Return the model identity, storage directory and number of loaded
    categories."""
    return {
        "name": settings.model_name,
        "version": settings.model_version,
        "model_dir": str(settings.model_dir),
        "categories": len(request.app.state.models),
    }
