from typing import TYPE_CHECKING

from dynamiq.utils.lazy_imports import lazy_exports

__all__ = [
    "InMemoryLongTermMemoryBackend",
    "PineconeLongTermMemoryBackend",
    "PostgresLongTermMemoryBackend",
    "QdrantLongTermMemoryBackend",
    "WeaviateLongTermMemoryBackend",
]

if TYPE_CHECKING:
    from dynamiq.memory.long_term.backends.in_memory import InMemoryLongTermMemoryBackend
    from dynamiq.memory.long_term.backends.pgvector import PostgresLongTermMemoryBackend
    from dynamiq.memory.long_term.backends.pinecone import PineconeLongTermMemoryBackend
    from dynamiq.memory.long_term.backends.qdrant import QdrantLongTermMemoryBackend
    from dynamiq.memory.long_term.backends.weaviate import WeaviateLongTermMemoryBackend

_LAZY_EXPORTS = {
    "InMemoryLongTermMemoryBackend": ".in_memory",
    "PostgresLongTermMemoryBackend": ".pgvector",
    "PineconeLongTermMemoryBackend": ".pinecone",
    "QdrantLongTermMemoryBackend": ".qdrant",
    "WeaviateLongTermMemoryBackend": ".weaviate",
}

lazy_exports(__name__, _LAZY_EXPORTS, globals())
