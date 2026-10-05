"""HTTP routes, one module per topic."""

from fastapi import APIRouter

from api.routes import health, models, predictions

router = APIRouter()
for module in (health, models, predictions):
    router.include_router(module.router)
