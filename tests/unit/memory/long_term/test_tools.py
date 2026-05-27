import pytest

from dynamiq.memory.long_term import LongTermMemory
from dynamiq.memory.long_term.backends.in_memory import InMemoryLongTermMemoryBackend
from dynamiq.nodes.tools.long_term_memory import RecallFactsTool, RememberFactTool, build_long_term_memory_tools


@pytest.fixture
def ltm(fake_embedder):
    return LongTermMemory(backend=InMemoryLongTermMemoryBackend(), embedder=fake_embedder)


# --- RememberFactTool ---


def test_remember_tool_persists_a_fact(ltm, user_id):
    tool = RememberFactTool(long_term_memory=ltm, user_id=user_id)
    result = tool.execute(tool.input_schema(content="User likes pizza"))
    fact_id = result["content"]["fact_id"]
    assert result["content"]["outcome"] == "created"
    assert ltm.get(fact_id).content == "User likes pizza"


def test_remember_tool_input_schema_has_no_user_id():
    """LLM-visible signature must not contain user_id — it's instance state."""
    assert "user_id" not in RememberFactTool.input_schema.model_fields
    assert {"content", "metadata"} <= set(RememberFactTool.input_schema.model_fields)


def test_remember_tool_uses_construction_user_id(ltm, user_id):
    tool = RememberFactTool(long_term_memory=ltm, user_id=user_id)
    result = tool.execute(tool.input_schema(content="x"))
    fact = ltm.get(result["content"]["fact_id"])
    assert fact.user_id == user_id


def test_remember_tool_idempotent_on_duplicate(ltm, user_id):
    tool = RememberFactTool(long_term_memory=ltm, user_id=user_id)
    a = tool.execute(tool.input_schema(content="x"))
    b = tool.execute(tool.input_schema(content="x"))
    assert a["content"]["fact_id"] == b["content"]["fact_id"]
    assert b["content"]["outcome"] == "unchanged"


def test_remember_tool_accepts_metadata(ltm, user_id):
    tool = RememberFactTool(long_term_memory=ltm, user_id=user_id)
    result = tool.execute(tool.input_schema(
        content="x", metadata={"category": "preference"}))
    fact = ltm.get(result["content"]["fact_id"])
    assert fact.metadata == {"category": "preference"}


def test_remember_tool_agent_optimized_returns_status_string(ltm, user_id):
    """Agent-mode output is a short human-readable status, not a dict."""
    tool = RememberFactTool(long_term_memory=ltm, user_id=user_id)
    tool.is_optimized_for_agents = True

    created = tool.execute(tool.input_schema(content="User likes pizza"))
    assert created["content"] == "Fact saved."

    unchanged = tool.execute(tool.input_schema(content="User likes pizza"))
    assert unchanged["content"] == "Already remembered."


def test_remember_tool_agent_optimized_reports_update(fake_embedder, user_id):
    """Agent-mode upsert renders as 'Fact updated.'"""
    ltm = LongTermMemory(
        backend=InMemoryLongTermMemoryBackend(),
        embedder=fake_embedder,
        upsert_threshold=0.0,
    )
    tool = RememberFactTool(long_term_memory=ltm, user_id=user_id)
    tool.is_optimized_for_agents = True

    tool.execute(tool.input_schema(content="User likes pizza"))
    updated = tool.execute(tool.input_schema(content="User loves pizza"))
    assert updated["content"] == "Fact updated."


# --- RecallFactsTool ---


def test_recall_tool_returns_hits(ltm, user_id):
    ltm.remember(content="User likes pizza", user_id=user_id)
    ltm.remember(content="User likes Python", user_id=user_id)
    tool = RecallFactsTool(long_term_memory=ltm, user_id=user_id)
    result = tool.execute(tool.input_schema(query="pizza", limit=2))
    items = result["content"]
    assert len(items) == 2
    for item in items:
        assert {"content", "score"} <= set(item.keys())
    scores = [it["score"] for it in items]
    assert scores == sorted(scores, reverse=True)


def test_recall_tool_input_schema_has_no_user_id():
    assert "user_id" not in RecallFactsTool.input_schema.model_fields
    assert {"query", "limit"} <= set(RecallFactsTool.input_schema.model_fields)


def test_recall_tool_isolates_users(ltm, user_id, other_user_id):
    ltm.remember(content="A's fact", user_id=user_id)
    ltm.remember(content="B's fact", user_id=other_user_id)
    tool = RecallFactsTool(long_term_memory=ltm, user_id=user_id)
    result = tool.execute(tool.input_schema(query="fact", limit=5))
    contents = {item["content"] for item in result["content"]}
    assert contents == {"A's fact"}


def test_recall_tool_empty_store_returns_empty(ltm, user_id):
    tool = RecallFactsTool(long_term_memory=ltm, user_id=user_id)
    result = tool.execute(tool.input_schema(query="anything"))
    assert result["content"] == []


def test_recall_tool_agent_optimized_returns_bullet_list(ltm, user_id):
    ltm.remember(content="User likes pizza", user_id=user_id)
    ltm.remember(content="User likes Python", user_id=user_id)
    tool = RecallFactsTool(long_term_memory=ltm, user_id=user_id)
    tool.is_optimized_for_agents = True
    result = tool.execute(tool.input_schema(query="pizza", limit=2))
    assert isinstance(result["content"], str)
    assert "- User likes pizza" in result["content"]
    assert "- User likes Python" in result["content"]


def test_recall_tool_agent_optimized_empty_message(ltm, user_id):
    tool = RecallFactsTool(long_term_memory=ltm, user_id=user_id)
    tool.is_optimized_for_agents = True
    result = tool.execute(tool.input_schema(query="anything"))
    assert result["content"] == "No relevant facts."


# --- factory ---


def test_factory_builds_default_two_tools(ltm, user_id):
    tools = build_long_term_memory_tools(long_term_memory=ltm, user_id=user_id)
    assert {t.name for t in tools} == {"remember_fact", "recall_facts"}


def test_factory_respects_include(ltm, user_id):
    tools = build_long_term_memory_tools(
        long_term_memory=ltm, user_id=user_id, include=("recall",),
    )
    assert [t.name for t in tools] == ["recall_facts"]


def test_factory_bakes_user_id_into_each_tool(ltm, user_id):
    tools = build_long_term_memory_tools(long_term_memory=ltm, user_id=user_id)
    for tool in tools:
        assert tool.user_id == user_id


def test_factory_ignores_unknown_include_keys(ltm, user_id):
    tools = build_long_term_memory_tools(
        long_term_memory=ltm, user_id=user_id,
        include=("recall", "unknown", "forget"),
    )
    assert [t.name for t in tools] == ["recall_facts"]


# --- serialization ---


def test_remember_tool_to_dict_round_trips_long_term_memory(ltm, user_id):
    """`to_dict` must not auto-dump `long_term_memory` (it holds runtime clients).

    The default `model_dump` would try to JSON-encode the embedder's connection
    and the backend's live client, blowing up tracing callbacks. The tool base
    excludes the field and re-adds it via `LongTermMemory.to_dict()`.
    """
    tool = RememberFactTool(long_term_memory=ltm, user_id=user_id)
    data = tool.to_dict()
    assert "long_term_memory" in data
    ltm_dump = data["long_term_memory"]
    assert isinstance(ltm_dump, dict)
    assert "backend" in ltm_dump and isinstance(ltm_dump["backend"], dict)
    assert "embedder" in ltm_dump and isinstance(ltm_dump["embedder"], dict)


def test_remember_tool_to_dict_accepts_include_secure_params(ltm, user_id):
    """`include_secure_params=True` must propagate through tool → LTM → backend → connection
    without raising. Connection.to_dict swallows the kwarg; backends pass it through."""
    tool = RememberFactTool(long_term_memory=ltm, user_id=user_id)
    data = tool.to_dict(include_secure_params=True)
    assert "long_term_memory" in data
    assert "backend" in data["long_term_memory"]
