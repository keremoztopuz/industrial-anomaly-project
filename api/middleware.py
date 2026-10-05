"""ASGI middleware that bounds the size of every request body."""

from fastapi.responses import JSONResponse

from api import config


class BodyTooLarge(Exception):
    """The request body is bigger than the configured limit."""


def declared_length_error(scope, limit):
    """Return an error response for a bad or oversized Content-Length header, else None."""
    for name, value in scope.get("headers", []):
        if name.lower() != b"content-length":
            continue
        try:
            length = int(value)
        except ValueError:
            length = -1
        if length < 0:
            return JSONResponse({"detail": "Invalid Content-Length"}, 400)
        if length > limit:
            return JSONResponse({"detail": "Request exceeds the byte limit"}, 413)
    return None


async def read_body(receive, limit):
    """Read the whole body, chunked or not; None means the client disconnected."""
    body = bytearray()
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            return None
        chunk = message.get("body", b"")
        if len(body) + len(chunk) > limit:
            raise BodyTooLarge
        body.extend(chunk)
        if not message.get("more_body", False):
            return bytes(body)


def replay(body, receive):
    """Hand the buffered body to the app once, then fall back to the real receive."""
    pending = [body]

    async def receive_buffered():
        if not pending:
            return await receive()
        return {"type": "http.request", "body": pending.pop(), "more_body": False}

    return receive_buffered


class RequestSizeLimit:
    """Check the complete body before multipart parsing, including chunked uploads."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        limit = config.MAX_REQUEST_BYTES
        error, body = declared_length_error(scope, limit), None
        if error is None:
            try:
                # Held in memory, which the limit bounds to about 11 MB per request.
                body = await read_body(receive, limit)
            except BodyTooLarge:
                error = JSONResponse({"detail": "Request exceeds the byte limit"}, 413)
        if error is not None:
            return await error(scope, receive, send)
        if body is None:
            return None
        return await self.app(scope, replay(body, receive), send)
