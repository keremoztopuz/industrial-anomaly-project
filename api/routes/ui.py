"""Test page navigation."""

from fastapi import APIRouter
from fastapi.responses import RedirectResponse

router = APIRouter()


@router.get("/ui", include_in_schema=False)
def redirect_to_test_page():
    """Keep the public scheme and path prefix behind a TLS proxy."""
    return RedirectResponse("./ui/")
