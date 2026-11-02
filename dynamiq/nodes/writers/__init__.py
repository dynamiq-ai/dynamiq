from typing import TYPE_CHECKING

from dynamiq.utils.lazy_imports import lazy_exports

if TYPE_CHECKING:
    from .chroma import ChromaDocumentWriter
    from .elasticsearch import ElasticsearchDocumentWriter
    from .milvus import MilvusDocumentWriter
    from .opensearch import OpenSearchDocumentWriter
    from .pgvector import PGVectorDocumentWriter
    from .pinecone import PineconeDocumentWriter
    from .qdrant import QdrantDocumentWriter
    from .turbopuffer import TurbopufferDocumentWriter
    from .weaviate import WeaviateDocumentWriter
    from .writer import VectorStoreWriter

_LAZY_EXPORTS = {
    "ChromaDocumentWriter": ".chroma",
    "ElasticsearchDocumentWriter": ".elasticsearch",
    "MilvusDocumentWriter": ".milvus",
    "OpenSearchDocumentWriter": ".opensearch",
    "PGVectorDocumentWriter": ".pgvector",
    "PineconeDocumentWriter": ".pinecone",
    "QdrantDocumentWriter": ".qdrant",
    "TurbopufferDocumentWriter": ".turbopuffer",
    "WeaviateDocumentWriter": ".weaviate",
    "VectorStoreWriter": ".writer",
}

lazy_exports(__name__, _LAZY_EXPORTS, globals())

__all__ = [
    "ChromaDocumentWriter",
    "ElasticsearchDocumentWriter",
    "MilvusDocumentWriter",
    "OpenSearchDocumentWriter",
    "PGVectorDocumentWriter",
    "PineconeDocumentWriter",
    "QdrantDocumentWriter",
    "TurbopufferDocumentWriter",
    "WeaviateDocumentWriter",
    "VectorStoreWriter",
]
