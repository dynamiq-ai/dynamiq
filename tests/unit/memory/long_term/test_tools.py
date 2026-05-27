"""Tests for the three long-term memory tools and the factory.

These tests do not invoke an LLM — they exercise the Node `execute()`
method directly, treating the tool the same way Agent's tool-use loop
would after the model emits a tool call.
"""
import pytest

from dynamiq.memory.long_term import LongTermMemory
from dynamiq.memory.long_term.backends.in_memory import InMemoryLongTermMemoryBackend
from dynamiq.nodes.tools.long_term_memory import (
    ForgetFactTool,
    RecallFactsTool,
    RememberFactTool,
    build_long_term_memory_tools,
)


@pytest.fixture
def ltm(fake_embedder):
    return LongTermMemory(backend=InMemoryLongTermMemoryBackend(), embedder=fake_embedder)


# --- RememberFactTool ---


def test_remember_tool_persists_a_fact(ltm, user_id):
    tool = RememberFactTool(long_term_memory=ltm, user_id=user_id)
    result = tool.execute(tool.input_schema(content="User likes pizza"))
    fact_id = result["content"]["fact_id"]
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


def test_remember_tool_accepts_metadata(ltm, user_id):
    tool = RememberFactTool(long_term_memory=ltm, user_id=user_id)
    result = tool.execute(tool.input_schema(
        content="x", metadata={"category": "preference"}))
    fact = ltm.get(result["content"]["fact_id"])
    assert fact.metadata == {"category": "preference"}


# --- RecallFactsTool ---


def test_recall_tool_returns_hits(ltm, user_id):
    ltm.remember(content="User likes pizza", user_id=user_id)
    ltm.remember(content="User likes Python", user_id=user_id)
    tool = RecallFactsTool(long_term_memory=ltm, user_id=user_id)
    result = tool.execute(tool.input_schema(query="pizza", limit=2))
    items = result["content"]
    assert len(items) == 2
    for item in items:
        assert {"fact_id", "content", "score"} <= set(item.keys())
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


# --- ForgetFactTool ---


def test_forget_tool_deletes_owned_fact(ltm, user_id):
    fact = ltm.remember(content="x", user_id=user_id)
    tool = ForgetFactTool(long_term_memory=ltm, user_id=user_id)
    result = tool.execute(tool.input_schema(fact_id=fact.id))
    assert result["content"]["status"] == "deleted"
    assert ltm.get(fact.id) is None


def test_forget_tool_returns_not_found_for_unknown_id(ltm, user_id):
    tool = ForgetFactTool(long_term_memory=ltm, user_id=user_id)
    result = tool.execute(tool.input_schema(fact_id="does-not-exist"))
    assert result["content"]["status"] == "not_found"


def test_forget_tool_returns_forbidden_on_cross_user(ltm, user_id, other_user_id):
    fact = ltm.remember(content="x", user_id=user_id)
    attacker = ForgetFactTool(long_term_memory=ltm, user_id=other_user_id)
    result = attacker.execute(attacker.input_schema(fact_id=fact.id))
    assert result["content"]["status"] == "forbidden"
    assert ltm.get(fact.id) is not None


def test_forget_tool_input_schema_has_no_user_id():
    assert "user_id" not in ForgetFactTool.input_schema.model_fields
    assert "fact_id" in ForgetFactTool.input_schema.model_fields


# --- factory ---


def test_factory_builds_all_three_by_default(ltm, user_id):
    tools = build_long_term_memory_tools(long_term_memory=ltm, user_id=user_id)
    assert {t.name for t in tools} == {"remember_fact", "recall_facts", "forget_fact"}


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
        include=("recall", "unknown"),
    )
    assert [t.name for t in tools] == ["recall_facts"]
