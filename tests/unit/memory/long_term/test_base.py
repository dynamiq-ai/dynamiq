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


def test_recall_with_zero_or_negative_limit_short_circuits(fake_embedder):
    """The base-class guard must return [] without embedding or searching —
    unified across all backends so each one need not re-implement it."""
    from dynamiq.memory.long_term.backends.in_memory import InMemoryLongTermMemoryBackend

    class _TrackingBackend(InMemoryLongTermMemoryBackend):
        search_calls: int = 0

        def search(self, *, query_embedding, scope, limit):
            type(self).search_calls += 1
            return super().search(query_embedding=query_embedding, scope=scope, limit=limit)

    _TrackingBackend.search_calls = 0
    backend = _TrackingBackend(embedder=fake_embedder)
    assert backend.recall(query="x", user_id="u1", limit=0) == []
    assert backend.recall(query="x", user_id="u1", limit=-3) == []
    assert _TrackingBackend.search_calls == 0


def test_list_all_with_zero_or_negative_limit_short_circuits(fake_embedder):
    from dynamiq.memory.long_term.backends.in_memory import InMemoryLongTermMemoryBackend

    backend = InMemoryLongTermMemoryBackend(embedder=fake_embedder)
    backend.remember(content="a", user_id="u1")
    assert backend.list_all(user_id="u1", limit=0) == []
    assert backend.list_all(user_id="u1", limit=-1) == []


def test_clear_user_rejects_empty_user_id(fake_embedder):
    """`clear_user("")` would resolve to `delete_scope({"user_id": ""})` and only
    spare facts with empty-string user_ids — block it at the public API."""
    from dynamiq.memory.long_term.backends.in_memory import InMemoryLongTermMemoryBackend

    backend = InMemoryLongTermMemoryBackend(embedder=fake_embedder)
    with pytest.raises(Exception, match="non-empty user_id"):
        backend.clear_user(user_id="")
