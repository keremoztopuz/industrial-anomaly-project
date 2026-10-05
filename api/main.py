"""Run the API with python -m api.main."""

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from anomaly.model.thresholds import load_thresholds
from api import config
from api.middleware import RequestSizeLimit
from api.routes import router
from api.services import model_loading


@asynccontextmanager
async def lifespan(app):
    app.state.models = model_loading.load_models(config.MODEL_DIR)
    app.state.thresholds = load_thresholds(config.MODEL_DIR / "thresholds.json")
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(RequestSizeLimit)
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host=config.HOST, port=config.PORT)
