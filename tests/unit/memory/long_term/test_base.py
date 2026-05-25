"""Tests for LongTermMemoryBackend ABC."""
import pytest

from dynamiq.memory.long_term.base import LongTermMemoryBackend


def test_long_term_memory_backend_is_abstract():
    with pytest.raises(TypeError):
        LongTermMemoryBackend()


def test_long_term_memory_backend_update_default_raises():
    """Phase 2 reserves `update`; v1 default raises NotImplementedError."""

    class TinyBackend(LongTermMemoryBackend):
        def insert(self, fact, embedding): ...
        def get(self, fact_id): return None
        def get_by_hash(self, *, user_id, content_hash): return None
        def delete(self, fact_id): ...
        def search(self, *, query_embedding, scope, limit): return []
        def list_by_scope(self, scope, limit=100): return []
        def delete_scope(self, scope): return 0

    backend = TinyBackend()
    with pytest.raises(NotImplementedError, match="Phase 2"):
        backend.update("f1", "x", [0.0])
