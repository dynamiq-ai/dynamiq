from .extractors import ByIndexExtractor, ByRegexExtractor, FileTypeExtractor

# Back-compat: the knowledge-graph nodes moved to ``dynamiq.nodes.knowledge_graph``. Resolve the old
# package-level names lazily (PEP 562) so old serialized ``type`` strings such as
# ``dynamiq.nodes.extractors.EntityExtractor`` still deserialize, without importing the graphs
# package at init time (which would create an import cycle via the agents/tools/converters chain).
_MOVED_TO_GRAPHS = {
    "EntityExtractor": "KnowledgeGraphEntityExtractor",
    "KnowledgeGraphWriter": "KnowledgeGraphWriter",
    "Ontology": "Ontology",
    "Triple": "Triple",
}


def __getattr__(name):
    if name in _MOVED_TO_GRAPHS:
        import dynamiq.nodes.knowledge_graph as _graphs

        return getattr(_graphs, _MOVED_TO_GRAPHS[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
