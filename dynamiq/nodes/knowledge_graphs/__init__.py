from typing import TYPE_CHECKING

from dynamiq.utils.lazy_imports import lazy_exports

__all__ = [
    "GraphNode",
    "GraphNodeRel",
    "GraphRelationship",
    "KnowledgeGraphEntityExtractor",
    "KnowledgeGraphRetriever",
    "KnowledgeGraphWriter",
    "Ontology",
    "Triple",
]

if TYPE_CHECKING:
    from .entity_extractor import (
        GraphNode,
        GraphNodeRel,
        GraphRelationship,
        KnowledgeGraphEntityExtractor,
        Ontology,
        Triple,
    )
    from .retriever import KnowledgeGraphRetriever
    from .writer import KnowledgeGraphWriter

_LAZY_EXPORTS = {
    "GraphNode": ".entity_extractor",
    "GraphNodeRel": ".entity_extractor",
    "GraphRelationship": ".entity_extractor",
    "KnowledgeGraphEntityExtractor": ".entity_extractor",
    "Ontology": ".entity_extractor",
    "Triple": ".entity_extractor",
    "KnowledgeGraphRetriever": ".retriever",
    "KnowledgeGraphWriter": ".writer",
}

lazy_exports(__name__, _LAZY_EXPORTS, globals())
