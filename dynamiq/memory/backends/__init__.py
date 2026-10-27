from typing import TYPE_CHECKING

from dynamiq.utils.lazy_imports import lazy_exports

if TYPE_CHECKING:
    from .base import MemoryBackend
    from .dynamiq import Dynamiq
    from .dynamo_db import DynamoDB
    from .in_memory import InMemory
    from .pinecone import Pinecone
    from .postgresql import PostgreSQL
    from .qdrant import Qdrant
    from .sqlite import SQLite
    from .weaviate import Weaviate

_LAZY_EXPORTS = {
    "MemoryBackend": ".base",
    "DynamoDB": ".dynamo_db",
    "InMemory": ".in_memory",
    "Dynamiq": ".dynamiq",
    "Pinecone": ".pinecone",
    "PostgreSQL": ".postgresql",
    "Qdrant": ".qdrant",
    "SQLite": ".sqlite",
    "Weaviate": ".weaviate",
}

lazy_exports(__name__, _LAZY_EXPORTS, globals())

__all__ = [
    "MemoryBackend",
    "DynamoDB",
    "InMemory",
    "Dynamiq",
    "Pinecone",
    "PostgreSQL",
    "Qdrant",
    "SQLite",
    "Weaviate",
]
