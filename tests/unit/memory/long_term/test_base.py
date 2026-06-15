import pytest

from dynamiq.memory.long_term.base import LongTermMemoryBackend


def test_long_term_memory_backend_is_abstract():
    with pytest.raises(TypeError):
        LongTermMemoryBackend()


def test_long_term_memory_backend_update_is_abstract():
    """Subclasses must implement `update` — semantic upsert depends on it."""

    class MissingUpdate(LongTermMemoryBackend):
        def insert(self, fact, embedding): ...
        def get(self, fact_id): return None
        def get_by_hash(self, *, user_id, content_hash): return None
        def delete(self, fact_id): ...
        def search(self, *, query_embedding, scope, limit): return []
        def list_by_scope(self, scope, limit=100): return []
        def delete_scope(self, scope): return 0

    with pytest.raises(TypeError, match="abstract"):
        MissingUpdate()


def test_guarded_ensure_calls_once_across_ops(fake_embedder):
    """`_ensure_storage` must fire exactly once across multiple high-level ops —
    backends rely on this to avoid hammering schema-create on every call."""
    from dynamiq.memory.long_term.backends.in_memory import InMemoryLongTermMemoryBackend

    class _CountingBackend(InMemoryLongTermMemoryBackend):
        ensure_calls: int = 0

        def _ensure_storage(self) -> None:
            type(self).ensure_calls += 1

    _CountingBackend.ensure_calls = 0
    backend = _CountingBackend(embedder=fake_embedder)
    backend.remember(content="a", user_id="u1")
    backend.remember(content="b", user_id="u1")
    backend.recall(query="a", user_id="u1")
    backend.list_all(user_id="u1")
    backend.forget(fact_id="nope", user_id="u1")
    assert _CountingBackend.ensure_calls == 1


def test_guarded_ensure_retries_after_failure(fake_embedder):
    """If `_ensure_storage` raises, the next op must retry — a transient network
    blip on schema-create shouldn't permanently disable the backend."""
    from dynamiq.memory.long_term.backends.in_memory import InMemoryLongTermMemoryBackend

    class _FlakyBackend(InMemoryLongTermMemoryBackend):
        attempts: int = 0

        def _ensure_storage(self) -> None:
            type(self).attempts += 1
            if type(self).attempts == 1:
                raise RuntimeError("transient")

    _FlakyBackend.attempts = 0
    backend = _FlakyBackend(embedder=fake_embedder)
    with pytest.raises(Exception):
        backend.remember(content="a", user_id="u1")
    backend.remember(content="a", user_id="u1")  # second attempt succeeds
    assert _FlakyBackend.attempts == 2
