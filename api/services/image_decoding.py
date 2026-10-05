"""Decode an uploaded image with byte, format and pixel limits."""

from io import BytesIO

from PIL import Image

from api.config import settings


class ImageTooLarge(ValueError):
    """The encoded file or decoded dimensions exceed the configured limit."""


def decode_image(file):
    contents = file.read(settings.max_upload_bytes + 1)
    if len(contents) > settings.max_upload_bytes:
        raise ImageTooLarge("Image file exceeds the byte limit")
    try:
        with Image.open(BytesIO(contents), formats=("PNG", "JPEG")) as image:
            if image.width * image.height > settings.max_image_pixels:
                raise ImageTooLarge("Image exceeds the pixel limit")
            return image.convert("RGB")
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise ImageTooLarge("Image exceeds the pixel limit") from error
    except (OSError, SyntaxError) as error:
        raise ValueError("Invalid image file") from error
