from .extractors import ByIndexExtractor, ByRegexExtractor, FileTypeExtractor

# Back-compat: the knowledge-graph nodes moved to ``dynamiq.nodes.knowledge_graphs``. Resolve the old
# package-level names lazily (PEP 562) so old serialized ``type`` strings such as
# ``dynamiq.nodes.extractors.EntityExtractor`` still deserialize, without importing the graphs
# package at init time (which would create an import cycle via the agents/tools/converters chain).
_MOVED_TO_GRAPHS = {
    "EntityExtractor": "KnowledgeGraphEntityExtractor",
    "KnowledgeGraphWriter": "KnowledgeGraphWriter",
    "Ontology": "Ontology",
    "Triple": "Triple",
}

# The moved names resolve via __getattr__ (below), so they are NOT in this module's __dict__.
# ``from ... import *`` skips __getattr__ when no __all__ is defined (it reads __dict__), which would
# drop the moved names. Listing them in __all__ makes ``import *`` do a real getattr per name, which
# DOES invoke __getattr__ -- restoring the star-import export while keeping the lazy, cycle-safe load.
__all__ = [
    "ByIndexExtractor",
    "ByRegexExtractor",
    "FileTypeExtractor",
    *_MOVED_TO_GRAPHS,
]


def __getattr__(name):
    if name in _MOVED_TO_GRAPHS:
        import dynamiq.nodes.knowledge_graphs as _graphs

        return getattr(_graphs, _MOVED_TO_GRAPHS[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
