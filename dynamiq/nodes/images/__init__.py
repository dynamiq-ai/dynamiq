from typing import TYPE_CHECKING

from dynamiq.utils.lazy_imports import lazy_exports

if TYPE_CHECKING:
    from .edit import ImageEdit
    from .generation import ImageGeneration, ImageResponseFormat, ImageSize

_LAZY_EXPORTS = {
    "ImageEdit": ".edit",
    "ImageGeneration": ".generation",
    "ImageResponseFormat": ".generation",
    "ImageSize": ".generation",
}

lazy_exports(__name__, _LAZY_EXPORTS, globals())

__all__ = [
    "ImageEdit",
    "ImageGeneration",
    "ImageResponseFormat",
    "ImageSize",
]
