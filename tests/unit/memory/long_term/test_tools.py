from dynamiq.memory.long_term.backends.in_memory import InMemoryLongTermMemoryBackend
from dynamiq.nodes.tools.long_term_memory import RecallFactsTool, RememberFactTool, build_long_term_memory_tools


# --- RememberFactTool ---


def test_remember_tool_persists_a_fact(backend, user_id):
    tool = RememberFactTool(backend=backend, user_id=user_id)
    result = tool.execute(tool.input_schema(content="User likes pizza"))
    fact_id = result["content"]["fact_id"]
    assert result["content"]["outcome"] == "created"
    assert backend.get(fact_id).content == "User likes pizza"


def test_remember_tool_input_schema_has_no_user_id():
    """LLM-visible signature must not contain user_id — it's instance state."""
    assert "user_id" not in RememberFactTool.input_schema.model_fields
    assert {"content", "metadata"} <= set(RememberFactTool.input_schema.model_fields)


def test_remember_tool_uses_construction_user_id(backend, user_id):
    tool = RememberFactTool(backend=backend, user_id=user_id)
    result = tool.execute(tool.input_schema(content="x"))
    fact = backend.get(result["content"]["fact_id"])
    assert fact.user_id == user_id


def test_remember_tool_idempotent_on_duplicate(backend, user_id):
    tool = RememberFactTool(backend=backend, user_id=user_id)
    a = tool.execute(tool.input_schema(content="x"))
    b = tool.execute(tool.input_schema(content="x"))
    assert a["content"]["fact_id"] == b["content"]["fact_id"]
    assert b["content"]["outcome"] == "unchanged"


def test_remember_tool_accepts_metadata(backend, user_id):
    tool = RememberFactTool(backend=backend, user_id=user_id)
    result = tool.execute(tool.input_schema(content="x", metadata={"category": "preference"}))
    fact = backend.get(result["content"]["fact_id"])
    assert fact.metadata == {"category": "preference"}


def test_remember_tool_agent_optimized_returns_status_string(backend, user_id):
    """Agent-mode output is a short human-readable status, not a dict."""
    tool = RememberFactTool(backend=backend, user_id=user_id)
    tool.is_optimized_for_agents = True

    created = tool.execute(tool.input_schema(content="User likes pizza"))
    assert created["content"] == "Fact saved."

    unchanged = tool.execute(tool.input_schema(content="User likes pizza"))
    assert unchanged["content"] == "Already remembered."


def test_remember_tool_agent_optimized_reports_update(fake_embedder, user_id):
    """Agent-mode upsert renders as 'Fact updated.'"""
    backend = InMemoryLongTermMemoryBackend(embedder=fake_embedder, upsert_threshold=0.0)
    tool = RememberFactTool(backend=backend, user_id=user_id)
    tool.is_optimized_for_agents = True

    tool.execute(tool.input_schema(content="User likes pizza"))
    updated = tool.execute(tool.input_schema(content="User loves pizza"))
    assert updated["content"] == "Fact updated."


# --- RecallFactsTool ---


def test_recall_tool_returns_hits(backend, user_id):
    backend.remember(content="User likes pizza", user_id=user_id)
    backend.remember(content="User likes Python", user_id=user_id)
    tool = RecallFactsTool(backend=backend, user_id=user_id)
    result = tool.execute(tool.input_schema(queries=["pizza"], limit=2))
    items = result["content"]
    assert len(items) == 2
    for item in items:
        assert {"content", "score"} <= set(item.keys())
    scores = [it["score"] for it in items]
    assert scores == sorted(scores, reverse=True)


def test_recall_tool_input_schema_has_no_user_id():
    assert "user_id" not in RecallFactsTool.input_schema.model_fields
    assert {"queries", "limit"} <= set(RecallFactsTool.input_schema.model_fields)


def test_recall_tool_isolates_users(backend, user_id, other_user_id):
    backend.remember(content="A's fact", user_id=user_id)
    backend.remember(content="B's fact", user_id=other_user_id)
    tool = RecallFactsTool(backend=backend, user_id=user_id)
    result = tool.execute(tool.input_schema(queries=["fact"], limit=5))
    contents = {item["content"] for item in result["content"]}
    assert contents == {"A's fact"}


def test_recall_tool_empty_store_returns_empty(backend, user_id):
    tool = RecallFactsTool(backend=backend, user_id=user_id)
    result = tool.execute(tool.input_schema(queries=["anything"]))
    assert result["content"] == []


def test_recall_tool_agent_optimized_returns_bullet_list(backend, user_id):
    backend.remember(content="User likes pizza", user_id=user_id)
    backend.remember(content="User likes Python", user_id=user_id)
    tool = RecallFactsTool(backend=backend, user_id=user_id)
    tool.is_optimized_for_agents = True
    result = tool.execute(tool.input_schema(queries=["pizza"], limit=2))
    assert isinstance(result["content"], str)
    assert "- User likes pizza" in result["content"]
    assert "- User likes Python" in result["content"]


def test_recall_tool_agent_optimized_empty_message(backend, user_id):
    tool = RecallFactsTool(backend=backend, user_id=user_id)
    tool.is_optimized_for_agents = True
    result = tool.execute(tool.input_schema(queries=["anything"]))
    assert result["content"] == "No relevant facts."


def test_recall_tool_merges_multiple_queries_and_dedupes(backend, user_id):
    """Multiple phrasings hitting the same fact must yield one entry, not duplicates."""
    backend.remember(content="User likes pizza", user_id=user_id)
    backend.remember(content="User likes Python", user_id=user_id)
    tool = RecallFactsTool(backend=backend, user_id=user_id)
    result = tool.execute(
        tool.input_schema(queries=["pizza", "User likes pizza", "favourite food"], limit=5)
    )
    contents = [it["content"] for it in result["content"]]
    assert len(contents) == len(set(contents))
    assert "User likes pizza" in contents


def test_recall_tool_fans_queries_out_in_parallel(backend, user_id, monkeypatch):
    """Multi-query recall must run `backend.recall` concurrently."""
    import time

    backend.remember(content="User likes pizza", user_id=user_id)
    backend.remember(content="User likes Python", user_id=user_id)

    delay = 0.1
    real_recall = type(backend).recall

    def slow_recall(self, *, query, user_id, limit):
        time.sleep(delay)
        return real_recall(self, query=query, user_id=user_id, limit=limit)

    monkeypatch.setattr(type(backend), "recall", slow_recall)
    tool = RecallFactsTool(backend=backend, user_id=user_id)

    start = time.monotonic()
    tool.execute(tool.input_schema(queries=["pizza", "Python", "favourite food", "language"], limit=5))
    elapsed = time.monotonic() - start

    # Sequential would be 4*delay; allow up to 2.5x for parallel + scheduling slack.
    assert elapsed < 2.5 * delay, f"recall fanout regressed to sequential ({elapsed:.3f}s)"


def test_recall_tool_rejects_empty_queries_list():
    """min_length=1 on `queries` must reject an empty list at schema validation."""
    import pytest as _pytest

    with _pytest.raises(Exception):
        RecallFactsTool.input_schema(queries=[])


def test_recall_tool_rejects_whitespace_only_query():
    """A blank or whitespace-only entry must be caught at validation time, not
    when the backend raises mid-execute."""
    import pytest as _pytest

    with _pytest.raises(Exception):
        RecallFactsTool.input_schema(queries=["   "])
    with _pytest.raises(Exception):
        RecallFactsTool.input_schema(queries=["valid", ""])


def test_recall_tool_strips_query_whitespace(backend, user_id):
    """Surrounding whitespace must be stripped so leading/trailing spaces don't
    affect the embedding (or cause spurious cache misses)."""
    backend.remember(content="User likes pizza", user_id=user_id)
    tool = RecallFactsTool(backend=backend, user_id=user_id)
    result = tool.execute(tool.input_schema(queries=["  pizza  "]))
    assert result["content"], "stripped query should still match the stored fact"


# --- factory ---


def test_factory_builds_default_two_tools(backend, user_id):
    tools = build_long_term_memory_tools(backend=backend, user_id=user_id)
    assert {t.name for t in tools} == {"remember_fact", "recall_facts"}


def test_factory_bakes_user_id_into_each_tool(backend, user_id):
    tools = build_long_term_memory_tools(backend=backend, user_id=user_id)
    for tool in tools:
        assert tool.user_id == user_id


# --- serialization ---


def test_remember_tool_to_dict_excludes_live_backend(backend, user_id):
    """`to_dict` must not auto-dump `backend` (it holds runtime clients + embedder).

    The default `model_dump` would try to JSON-encode the embedder's connection
    and the backend's live client, blowing up tracing callbacks. The tool base
    excludes the field and re-adds it via `LongTermMemoryBackend.to_dict()`.
    """
    tool = RememberFactTool(backend=backend, user_id=user_id)
    data = tool.to_dict()
    assert "backend" in data
    backend_dump = data["backend"]
    assert isinstance(backend_dump, dict)
    assert "embedder" in backend_dump and isinstance(backend_dump["embedder"], dict)


def test_remember_tool_to_dict_accepts_include_secure_params(backend, user_id):
    """`include_secure_params=True` must propagate through tool → backend → connection
    without raising. Connection.to_dict swallows the kwarg; backends pass it through."""
    tool = RememberFactTool(backend=backend, user_id=user_id)
    data = tool.to_dict(include_secure_params=True)
    assert "backend" in data
    assert "embedder" in data["backend"]
