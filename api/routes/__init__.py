"""HTTP routes, one module per topic."""

from fastapi import APIRouter

from api.routes import health, models, predictions, ui

router = APIRouter()
for module in (health, models, predictions, ui):
    router.include_router(module.router)
