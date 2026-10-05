"""Run the API with python -m api.main."""

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from api import config, services
from api.routes import router


class RequestSizeLimit:
    """Check the complete body before multipart parsing, including chunked uploads."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        limit = config.MAX_REQUEST_BYTES
        for name, value in scope.get("headers", []):
            if name.lower() == b"content-length":
                try:
                    length = int(value)
                    if length < 0:
                        raise ValueError
                except ValueError:
                    return await JSONResponse({"detail": "Invalid Content-Length"}, 400)(
                        scope, receive, send)
                if length > limit:
                    return await JSONResponse({"detail": "Request exceeds the byte limit"}, 413)(
                        scope, receive, send)
        # ponytail: one bounded body per request; use a spool if concurrency makes RAM a concern.
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > limit:
                return await JSONResponse({"detail": "Request exceeds the byte limit"}, 413)(
                    scope, receive, send)
            body.extend(chunk)
            if not message.get("more_body", False):
                break

        async def replay():
            nonlocal body
            if body is None:
                return await receive()
            message = {"type": "http.request", "body": bytes(body), "more_body": False}
            body = None
            return message

        await self.app(scope, replay, send)


@asynccontextmanager
async def lifespan(app):
    app.state.models = services.load_models(config.MODEL_DIR)
    app.state.thresholds = services.load_thresholds(config.MODEL_DIR / "thresholds.json")
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(RequestSizeLimit)
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host=config.HOST, port=config.PORT)
