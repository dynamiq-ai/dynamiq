from typing import TYPE_CHECKING

from dynamiq.components.splitters.auto import AutoSplitterRule, AutoSplitterStrategy
from dynamiq.components.splitters.base import IdStrategy, LengthUnit
from dynamiq.components.splitters.code import CodeParser
from dynamiq.components.splitters.language import Language, get_separators_for_language
from dynamiq.components.splitters.semantic import BreakpointThresholdType
from dynamiq.utils.lazy_imports import lazy_exports

__all__ = [
    "AutoSplitter",
    "AutoSplitterRule",
    "AutoSplitterStrategy",
    "BreakpointThresholdType",
    "CodeParser",
    "CodeSplitter",
    "ContextualSplitter",
    "DocumentSplitter",
    "HTMLHeaderSplitter",
    "HTMLSectionSplitter",
    "IdStrategy",
    "Language",
    "LengthUnit",
    "MarkdownHeaderSplitter",
    "RecursiveCharacterSplitter",
    "RecursiveJsonSplitter",
    "SemanticSplitter",
    "TokenSplitter",
    "get_separators_for_language",
]

if TYPE_CHECKING:
    from .auto import AutoSplitter
    from .code import CodeSplitter
    from .contextual import ContextualSplitter
    from .document import DocumentSplitter
    from .html import HTMLHeaderSplitter, HTMLSectionSplitter
    from .json import RecursiveJsonSplitter
    from .markdown_header import MarkdownHeaderSplitter
    from .recursive_character import RecursiveCharacterSplitter
    from .semantic import SemanticSplitter
    from .token import TokenSplitter

_LAZY_EXPORTS = {
    "AutoSplitter": ".auto",
    "CodeSplitter": ".code",
    "ContextualSplitter": ".contextual",
    "DocumentSplitter": ".document",
    "HTMLHeaderSplitter": ".html",
    "HTMLSectionSplitter": ".html",
    "RecursiveJsonSplitter": ".json",
    "MarkdownHeaderSplitter": ".markdown_header",
    "RecursiveCharacterSplitter": ".recursive_character",
    "SemanticSplitter": ".semantic",
    "TokenSplitter": ".token",
}

lazy_exports(__name__, _LAZY_EXPORTS, globals())
