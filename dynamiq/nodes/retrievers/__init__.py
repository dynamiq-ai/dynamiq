from .chroma import ChromaDocumentRetriever
from .elasticsearch import ElasticsearchDocumentRetriever
from .milvus import MilvusDocumentRetriever
from .opensearch import OpenSearchDocumentRetriever
from .pgvector import PGVectorDocumentRetriever
from .pinecone import PineconeDocumentRetriever
from .qdrant import QdrantDocumentRetriever
from .retriever import VectorStoreRetriever
from .weaviate import WeaviateDocumentRetriever

# ``GraphRetriever`` resolves via __getattr__ (below), so it is NOT in this module's __dict__.
# ``from ... import *`` skips __getattr__ when no __all__ is defined (it reads __dict__), which would
# drop ``GraphRetriever``. Listing it in __all__ makes ``import *`` do a real getattr per name, which
# DOES invoke __getattr__ -- restoring the star-import export while keeping the lazy, cycle-safe load.
__all__ = [
    "ChromaDocumentRetriever",
    "ElasticsearchDocumentRetriever",
    "MilvusDocumentRetriever",
    "OpenSearchDocumentRetriever",
    "PGVectorDocumentRetriever",
    "PineconeDocumentRetriever",
    "QdrantDocumentRetriever",
    "VectorStoreRetriever",
    "WeaviateDocumentRetriever",
    "GraphRetriever",
]


def __getattr__(name):
    # Back-compat: GraphRetriever moved to dynamiq.nodes.knowledge_graph.KnowledgeGraphRetriever.
    # Resolved lazily so old serialized ``type`` strings (dynamiq.nodes.retrievers.GraphRetriever)
    # still deserialize without importing the graphs package at init time (avoids an import cycle).
    if name == "GraphRetriever":
        from dynamiq.nodes.knowledge_graph import KnowledgeGraphRetriever

        return KnowledgeGraphRetriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
