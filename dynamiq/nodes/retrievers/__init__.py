from typing import TYPE_CHECKING

from dynamiq.utils.lazy_imports import lazy_exports

__all__ = [
    "ChromaDocumentRetriever",
    "ElasticsearchDocumentRetriever",
    "MilvusDocumentRetriever",
    "OpenSearchDocumentRetriever",
    "PGVectorDocumentRetriever",
    "PineconeDocumentRetriever",
    "QdrantDocumentRetriever",
    "TurbopufferDocumentRetriever",
    "VectorStoreRetriever",
    "WeaviateDocumentRetriever",
    "GraphRetriever",
]

if TYPE_CHECKING:
    from .chroma import ChromaDocumentRetriever
    from .elasticsearch import ElasticsearchDocumentRetriever
    from .milvus import MilvusDocumentRetriever
    from .opensearch import OpenSearchDocumentRetriever
    from .pgvector import PGVectorDocumentRetriever
    from .pinecone import PineconeDocumentRetriever
    from .qdrant import QdrantDocumentRetriever
    from .retriever import VectorStoreRetriever
    from .turbopuffer import TurbopufferDocumentRetriever
    from .weaviate import WeaviateDocumentRetriever

_LAZY_EXPORTS = {
    # GraphRetriever moved to dynamiq.nodes.knowledge_graphs.KnowledgeGraphRetriever. The alias keeps old type
    # strings, such as dynamiq.nodes.retrievers.GraphRetriever, loading.
    "GraphRetriever": "dynamiq.nodes.knowledge_graphs:KnowledgeGraphRetriever",
    "ChromaDocumentRetriever": ".chroma",
    "ElasticsearchDocumentRetriever": ".elasticsearch",
    "MilvusDocumentRetriever": ".milvus",
    "OpenSearchDocumentRetriever": ".opensearch",
    "PGVectorDocumentRetriever": ".pgvector",
    "PineconeDocumentRetriever": ".pinecone",
    "QdrantDocumentRetriever": ".qdrant",
    "VectorStoreRetriever": ".retriever",
    "TurbopufferDocumentRetriever": ".turbopuffer",
    "WeaviateDocumentRetriever": ".weaviate",
}

lazy_exports(__name__, _LAZY_EXPORTS, globals())
