"""Liveness endpoints."""

from fastapi import APIRouter

from api.schemas import StatusResponse

router = APIRouter()


@router.get("/", response_model=StatusResponse, summary="API status")
def read_root():
    """Report that the HTTP API is running."""
    return {"status": "API is running"}


@router.get(
    "/health", response_model=StatusResponse, summary="Check HTTP liveness"
)
def check_liveness():
    """Report process liveness; models are validated during startup."""
    return {"status": "healthy"}
