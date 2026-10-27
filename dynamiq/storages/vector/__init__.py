from typing import TYPE_CHECKING

from dynamiq.utils.lazy_imports import lazy_exports

if TYPE_CHECKING:
    from .chroma import ChromaVectorStore
    from .elasticsearch import ElasticsearchVectorStore
    from .milvus import MilvusVectorStore
    from .opensearch import OpenSearchVectorStore
    from .pgvector import PGVectorStore
    from .pinecone import PineconeVectorStore
    from .qdrant import QdrantVectorStore
    from .turbopuffer import TurbopufferVectorStore
    from .weaviate import WeaviateVectorStore

_LAZY_EXPORTS = {
    "ChromaVectorStore": ".chroma",
    "ElasticsearchVectorStore": ".elasticsearch",
    "MilvusVectorStore": ".milvus",
    "OpenSearchVectorStore": ".opensearch",
    "PGVectorStore": ".pgvector",
    "PineconeVectorStore": ".pinecone",
    "QdrantVectorStore": ".qdrant",
    "TurbopufferVectorStore": ".turbopuffer",
    "WeaviateVectorStore": ".weaviate",
}

lazy_exports(__name__, _LAZY_EXPORTS, globals())

__all__ = [
    "ChromaVectorStore",
    "ElasticsearchVectorStore",
    "MilvusVectorStore",
    "OpenSearchVectorStore",
    "PGVectorStore",
    "PineconeVectorStore",
    "QdrantVectorStore",
    "TurbopufferVectorStore",
    "WeaviateVectorStore",
]
