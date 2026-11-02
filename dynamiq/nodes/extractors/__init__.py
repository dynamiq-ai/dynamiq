from typing import TYPE_CHECKING

from dynamiq.utils.lazy_imports import lazy_exports

if TYPE_CHECKING:
    from .extractors import ByIndexExtractor, ByRegexExtractor, FileTypeExtractor

# The knowledge graph nodes moved to dynamiq.nodes.knowledge_graphs. These aliases keep old type strings, such as
# dynamiq.nodes.extractors.EntityExtractor, loading.
_LAZY_EXPORTS = {
    "ByIndexExtractor": ".extractors",
    "ByRegexExtractor": ".extractors",
    "FileTypeExtractor": ".extractors",
    "EntityExtractor": "dynamiq.nodes.knowledge_graphs:KnowledgeGraphEntityExtractor",
    "KnowledgeGraphWriter": "dynamiq.nodes.knowledge_graphs:KnowledgeGraphWriter",
    "Ontology": "dynamiq.nodes.knowledge_graphs:Ontology",
    "Triple": "dynamiq.nodes.knowledge_graphs:Triple",
}

lazy_exports(__name__, _LAZY_EXPORTS, globals())

__all__ = [
    "ByIndexExtractor",
    "ByRegexExtractor",
    "FileTypeExtractor",
    "EntityExtractor",
    "KnowledgeGraphWriter",
    "Ontology",
    "Triple",
]
