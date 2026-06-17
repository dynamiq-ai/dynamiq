import hashlib
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import patch

import pytest

from dynamiq.connections import BaseConnection
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.memory.long_term import LongTermMemoryConfig
from dynamiq.memory.long_term.backends.in_memory import InMemoryLongTermMemoryBackend
from dynamiq.nodes.agents.base import Agent
from dynamiq.nodes.embedders.base import TextEmbedder, TextEmbedderInputSchema
from dynamiq.nodes.llms import OpenAI


class _StubConnection(BaseConnection):
    def connect(self) -> None:
        return None


class _FakeEmbedder(TextEmbedder):
    name: str = "fake-text-embedder"
    connection: BaseConnection = _StubConnection()
    DIM: ClassVar[int] = 16

    def execute(self, input_data: TextEmbedderInputSchema, config=None, **kwargs):
        text = input_data.query if hasattr(input_data, "query") else input_data["query"]
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [(b / 127.5) - 1.0 for b in digest[: self.DIM]]
        norm = sum(x * x for x in raw) ** 0.5 or 1.0
        return {"query": text, "embedding": [x / norm for x in raw]}


@pytest.fixture
def ltm():
    return LongTermMemoryConfig(backend=InMemoryLongTermMemoryBackend(embedder=_FakeEmbedder()))


@pytest.fixture
def llm():
    """Real OpenAI LLM object — never executed in these tests. Constructed
    only to satisfy Agent's pydantic validation."""
    return OpenAI(
        connection=OpenAIConnection(api_key="test-key"),
        model="gpt-4o",
    )


def _ltm_config(*, enabled=True) -> LongTermMemoryConfig:
    backend = InMemoryLongTermMemoryBackend(embedder=_FakeEmbedder())
    return LongTermMemoryConfig(backend=backend, enabled=enabled)


def _make_agent(llm, *, ltm=None) -> Agent:
    kwargs = {"name": "test", "llm": llm, "tools": []}
    if ltm is not None:
        kwargs["long_term_memory"] = ltm
    return Agent(**kwargs)


def _input(user_id=None, session_id=None):
    return SimpleNamespace(user_id=user_id, session_id=session_id, input="hi")


# --- LongTermMemoryConfig ---


def test_config_defaults_to_enabled():
    assert _ltm_config().enabled is True


# --- Agent field declarations ---


def test_agent_has_long_term_memory_field():
    fields = Agent.model_fields
    assert "long_term_memory" in fields
    assert fields["long_term_memory"].default is None


def test_agent_long_term_memory_defaults_to_none(llm):
    agent = _make_agent(llm)
    assert agent.long_term_memory is None


# --- _build_long_term_memory_tools ---


def test_build_returns_default_tools_when_ltm_and_user_id_present(llm, ltm):
    agent = _make_agent(llm, ltm=ltm)
    tools = agent._build_long_term_memory_tools(_input(user_id="u1"))
    assert {t.name for t in tools} == {"remember_fact", "recall_facts"}


def test_build_returns_empty_when_no_user_id(llm, ltm):
    agent = _make_agent(llm, ltm=ltm)
    assert agent._build_long_term_memory_tools(_input(session_id="s1")) == []


def test_build_returns_empty_when_no_long_term_memory(llm):
    agent = _make_agent(llm)
    assert agent._build_long_term_memory_tools(_input(user_id="u1")) == []


def test_build_returns_empty_when_disabled(llm):
    agent = _make_agent(llm, ltm=_ltm_config(enabled=False))
    assert agent._build_long_term_memory_tools(_input(user_id="u1")) == []


def test_build_bakes_user_id_into_each_tool(llm, ltm):
    agent = _make_agent(llm, ltm=ltm)
    tools = agent._build_long_term_memory_tools(_input(user_id="u1"))
    for tool in tools:
        assert tool.user_id == "u1"


def test_build_sets_is_optimized_for_agents_on_each_tool(llm, ltm):
    """LTM tools are built per-run, after `init_components` has run, so the agent
    must flip `is_optimized_for_agents` itself — otherwise remember/recall would
    return raw dicts instead of the friendly status strings the LLM expects."""
    agent = _make_agent(llm, ltm=ltm)
    tools = agent._build_long_term_memory_tools(_input(user_id="u1"))
    assert tools and all(t.is_optimized_for_agents for t in tools)


def test_function_calling_schemas_include_ltm_overlay(llm, ltm):
    """In FUNCTION_CALLING mode the per-call LTM tools must appear in the
    generated tool schemas, otherwise the LLM can never call remember/recall."""
    from dynamiq.nodes.agents.agent import Agent as ReActAgent
    from dynamiq.nodes.agents.base import _run_extra_tools
    from dynamiq.nodes.types import InferenceMode

    agent = ReActAgent(name="t", llm=llm, tools=[], long_term_memory=ltm, inference_mode=InferenceMode.FUNCTION_CALLING)
    base_tools, _ = agent._effective_inference_schemas()
    base_names = {schema["function"]["name"] for schema in (base_tools or [])}
    assert "remember_fact" not in base_names  # not present without an overlay

    ltm_tools = agent._build_long_term_memory_tools(_input(user_id="u1"))
    token = _run_extra_tools.set(ltm_tools)
    try:
        fc_tools, _ = agent._effective_inference_schemas()
    finally:
        _run_extra_tools.reset(token)

    names = {schema["function"]["name"] for schema in fc_tools}
    assert {"remember_fact", "recall_facts"} <= names


def test_xml_prompt_includes_tool_blocks_when_only_ltm_configured(llm, ltm):
    """In XML/ReAct mode the system prompt template must reserve tool blocks
    when LTM is the only source of tools — otherwise the per-call tool
    description has no placeholder and remember/recall stay invisible."""
    from dynamiq.nodes.agents.agent import Agent as ReActAgent
    from dynamiq.nodes.types import InferenceMode

    agent = ReActAgent(name="t", llm=llm, tools=[], long_term_memory=ltm, inference_mode=InferenceMode.XML)
    tools_block = agent.system_prompt_manager._prompt_blocks.get("tools", "")
    assert "{{ tool_description }}" in tools_block


def test_xml_prompt_omits_tool_blocks_when_ltm_disabled(llm):
    """Disabled LTM must not flip `has_tools` on — the template should still
    render the no-tools instructions when nothing else provides tools."""
    from dynamiq.nodes.agents.agent import Agent as ReActAgent
    from dynamiq.nodes.types import InferenceMode

    agent = ReActAgent(
        name="t",
        llm=llm,
        tools=[],
        long_term_memory=_ltm_config(enabled=False),
        inference_mode=InferenceMode.XML,
    )
    assert agent.system_prompt_manager._prompt_blocks.get("tools", "") == ""


def test_init_components_initializes_ltm_embedder(llm):
    """The embedder is a ConnectionNode whose `text_embedder` client is built
    during `init_components`; without that, the first recall AttributeErrors
    on a None client."""
    init_calls: list = []

    class _RecordingEmbedder(_FakeEmbedder):
        is_postponed_component_init: bool = True

        def init_components(self, connection_manager=None):
            init_calls.append(connection_manager)

    ltm_with_postponed = LongTermMemoryConfig(backend=InMemoryLongTermMemoryBackend(embedder=_RecordingEmbedder()))
    agent = _make_agent(llm, ltm=ltm_with_postponed)
    init_calls.clear()
    agent.init_components()
    assert len(init_calls) == 1


# --- per-call ContextVar overlay: LTM tools never mutate self.tools ---


def _patch_run_agent_capture_runtime_tools(agent, captured):
    """Capture what the LLM-facing `tool_by_names` resolution sees mid-run."""

    def fake_run(*args, **kwargs):
        captured.append(set(agent.tool_by_names.keys()))
        return "ok"

    return patch.object(agent, "_run_agent", side_effect=fake_run)


def test_execute_exposes_ltm_tools_during_run_only(llm, ltm):
    """LTM tools must be visible to the tool-resolution properties during the
    run, and absent from both `self.tools` and the properties after."""
    agent = _make_agent(llm, ltm=ltm)
    original_tools = list(agent.tools)
    captured: list[set[str]] = []

    with _patch_run_agent_capture_runtime_tools(agent, captured):
        agent.run_sync(input_data={"input": "hi", "user_id": "u1"})

    assert {"remember_fact", "recall_facts"} <= captured[0]
    assert agent.tools == original_tools
    assert {"remember_fact", "recall_facts"}.isdisjoint(agent.tool_by_names.keys())


def test_execute_clears_ltm_overlay_even_when_run_raises(llm, ltm):
    agent = _make_agent(llm, ltm=ltm)
    original_tools = list(agent.tools)

    with patch.object(agent, "_run_agent", side_effect=RuntimeError("boom")):
        agent.run_sync(input_data={"input": "hi", "user_id": "u1"})

    assert agent.tools == original_tools
    assert {"remember_fact", "recall_facts"}.isdisjoint(agent.tool_by_names.keys())


def test_execute_no_ltm_overlay_when_no_user_id(llm, ltm):
    agent = _make_agent(llm, ltm=ltm)
    captured: list[set[str]] = []

    with _patch_run_agent_capture_runtime_tools(agent, captured):
        agent.run_sync(input_data={"input": "hi"})

    assert {"remember_fact", "recall_facts"}.isdisjoint(captured[0])


def test_execute_no_ltm_overlay_when_no_long_term_memory(llm):
    agent = _make_agent(llm)
    captured: list[set[str]] = []

    with _patch_run_agent_capture_runtime_tools(agent, captured):
        agent.run_sync(input_data={"input": "hi", "user_id": "u1"})

    assert {"remember_fact", "recall_facts"}.isdisjoint(captured[0])


def test_execute_does_not_serialize_concurrent_calls_when_ltm_configured(llm, ltm):
    """With the ContextVar overlay, concurrent execute() calls on the same
    LTM-configured agent must run truly in parallel — no shared lock."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    agent = _make_agent(llm, ltm=ltm)
    barrier = threading.Barrier(2, timeout=5)

    def fake_run(*args, **kwargs):
        barrier.wait()  # times out under a lock; verifies real concurrency
        return "ok"

    with patch.object(agent, "_run_agent", side_effect=fake_run):
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(agent.run_sync, input_data={"input": "hi", "user_id": "u1"}),
                pool.submit(agent.run_sync, input_data={"input": "hi", "user_id": "u2"}),
            ]
            for f in futures:
                f.result(timeout=10)


def test_concurrent_calls_isolate_per_user_ltm_tools(llm, ltm):
    """Two concurrent execute() calls with different user_ids must each see
    only their own LTM tools via the per-task ContextVar overlay."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    agent = _make_agent(llm, ltm=ltm)
    snapshots: dict[str, set[str]] = {}
    snapshots_lock = threading.Lock()
    barrier = threading.Barrier(2, timeout=5)

    def fake_run(*args, **kwargs):
        barrier.wait()
        resolved = agent.tool_by_names
        bound_user_ids = {t.user_id for t in resolved.values() if hasattr(t, "user_id")}
        assert len(bound_user_ids) == 1, f"cross-user leakage: {bound_user_ids}"
        (uid,) = bound_user_ids
        with snapshots_lock:
            snapshots[uid] = {name for name, t in resolved.items() if hasattr(t, "user_id")}
        return "ok"

    with patch.object(agent, "_run_agent", side_effect=fake_run):
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(agent.run_sync, input_data={"input": "hi", "user_id": "u1"}),
                pool.submit(agent.run_sync, input_data={"input": "hi", "user_id": "u2"}),
            ]
            for f in futures:
                f.result(timeout=10)

    assert set(snapshots.keys()) == {"u1", "u2"}
    for tool_names in snapshots.values():
        assert tool_names == {"remember_fact", "recall_facts"}


def test_concurrent_no_user_id_call_does_not_see_other_users_ltm_tools(llm, ltm):
    """A concurrent no-user_id execute must not observe another call's
    user-scoped tools — ContextVar isolation guarantees this without a lock."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    agent = _make_agent(llm, ltm=ltm)
    snapshots: dict[str, set] = {}
    snapshots_lock = threading.Lock()
    barrier = threading.Barrier(2, timeout=5)

    def fake_run(*args, **kwargs):
        barrier.wait()
        resolved = agent.tool_by_names
        bound = {getattr(t, "user_id", None) for t in resolved.values() if hasattr(t, "user_id")}
        with snapshots_lock:
            key = next(iter(bound), "none")
            snapshots[key] = bound
        return "ok"

    with patch.object(agent, "_run_agent", side_effect=fake_run):
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(agent.run_sync, input_data={"input": "hi", "user_id": "u1"}),
                pool.submit(agent.run_sync, input_data={"input": "hi"}),
            ]
            for f in futures:
                f.result(timeout=10)

    assert snapshots.get("u1") == {"u1"}
    assert snapshots.get("none", set()) == set()


def test_sub_agent_without_ltm_does_not_inherit_parent_overlay(llm, ltm):
    """A sub-agent with no LTM of its own, executed inside a parent agent's
    LTM-active context (e.g. a child thread via ContextAwareThreadPoolExecutor),
    must NOT inherit the parent's user-scoped tools — even though ContextVars
    propagate to child contexts by default. The fix: every execute() sets the
    overlay unconditionally, even to an empty list."""
    parent = _make_agent(llm, ltm=ltm)
    sub_agent = _make_agent(llm)  # no LTM
    sub_captured: list[set[str]] = []

    def parent_run(*args, **kwargs):
        with _patch_run_agent_capture_runtime_tools(sub_agent, sub_captured):
            sub_agent.run_sync(input_data={"input": "hi", "user_id": "u1"})
        return "ok"

    with patch.object(parent, "_run_agent", side_effect=parent_run):
        parent.run_sync(input_data={"input": "hi", "user_id": "u1"})

    assert sub_captured, "sub-agent _run_agent never fired"
    assert {"remember_fact", "recall_facts"}.isdisjoint(sub_captured[0]), (
        "sub-agent saw parent's LTM tools via inherited ContextVar"
    )
