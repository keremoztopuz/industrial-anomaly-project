"""Run the API with python -m api.main."""

from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from anomaly.model.thresholds import load_thresholds
from api.config import settings
from api.middleware import RequestSizeLimit
from api.routes import router
from api.services import model_loading


@asynccontextmanager
async def lifespan(app):
    app.state.models = model_loading.load_models(settings.model_dir)
    app.state.thresholds = load_thresholds(
        settings.model_dir / "thresholds.json"
    )
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(RequestSizeLimit)
app.include_router(router)
app.mount(
    "/ui", StaticFiles(directory=Path(__file__).parent / "ui", html=True),
    name="ui",
)

if __name__ == "__main__":
    uvicorn.run(app, host=settings.host, port=settings.port)
