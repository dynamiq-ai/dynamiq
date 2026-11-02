from typing import TYPE_CHECKING

from dynamiq.utils.lazy_imports import lazy_exports

if TYPE_CHECKING:
    from .cohere import CohereReranker
    from .llm import LLMDocumentRanker
    from .recency import TimeWeightedDocumentRanker

_LAZY_EXPORTS = {
    "CohereReranker": ".cohere",
    "LLMDocumentRanker": ".llm",
    "TimeWeightedDocumentRanker": ".recency",
}

lazy_exports(__name__, _LAZY_EXPORTS, globals())

__all__ = [
    "CohereReranker",
    "LLMDocumentRanker",
    "TimeWeightedDocumentRanker",
]
