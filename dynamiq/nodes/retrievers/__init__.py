from .chroma import ChromaDocumentRetriever
from .elasticsearch import ElasticsearchDocumentRetriever
from .milvus import MilvusDocumentRetriever
from .opensearch import OpenSearchDocumentRetriever
from .pgvector import PGVectorDocumentRetriever
from .pinecone import PineconeDocumentRetriever
from .qdrant import QdrantDocumentRetriever
from .retriever import VectorStoreRetriever
from .weaviate import WeaviateDocumentRetriever


def __getattr__(name):
    # Back-compat: GraphRetriever moved to dynamiq.nodes.knowledge_graph.KnowledgeGraphRetriever.
    # Resolved lazily so old serialized ``type`` strings (dynamiq.nodes.retrievers.GraphRetriever)
    # still deserialize without importing the graphs package at init time (avoids an import cycle).
    if name == "GraphRetriever":
        from dynamiq.nodes.knowledge_graph import KnowledgeGraphRetriever

        return KnowledgeGraphRetriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
