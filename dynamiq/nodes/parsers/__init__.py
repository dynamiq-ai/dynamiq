from typing import TYPE_CHECKING

from dynamiq.utils.lazy_imports import lazy_exports

if TYPE_CHECKING:
    from .mistral_ocr import MistralOCR, MistralOCRInputSchema

_LAZY_EXPORTS = {
    "MistralOCR": ".mistral_ocr",
    "MistralOCRInputSchema": ".mistral_ocr",
}

lazy_exports(__name__, _LAZY_EXPORTS, globals())

__all__ = [
    "MistralOCR",
    "MistralOCRInputSchema",
]
