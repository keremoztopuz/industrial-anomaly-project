"""API settings: the shared settings plus the HTTP server and the upload limits."""

from pydantic import PositiveInt, model_validator

from anomaly.settings import Settings

REQUEST_OVERHEAD_BYTES = 1024 * 1024  # multipart headers, on top of the image bytes


class ApiSettings(Settings):
    host: str = "127.0.0.1"
    port: PositiveInt = 8000
    max_upload_bytes: PositiveInt = 10 * 1024 * 1024
    max_image_pixels: PositiveInt = 16_000_000
    max_request_bytes: PositiveInt | None = None

    @model_validator(mode="after")
    def default_request_limit(self):
        if self.max_request_bytes is None:
            self.max_request_bytes = self.max_upload_bytes + REQUEST_OVERHEAD_BYTES
        return self


settings = ApiSettings()
