from typing import TYPE_CHECKING

from dynamiq.utils.lazy_imports import lazy_exports

if TYPE_CHECKING:
    from .utils import Input, Output

_LAZY_EXPORTS = {
    "Input": ".utils",
    "Output": ".utils",
}

lazy_exports(__name__, _LAZY_EXPORTS, globals())

__all__ = [
    "Input",
    "Output",
]
