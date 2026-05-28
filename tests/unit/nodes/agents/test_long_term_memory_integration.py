import hashlib
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import patch

import pytest

from dynamiq.connections import BaseConnection
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.memory.long_term import LongTermMemory, LongTermMemoryConfig
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
    return LongTermMemory(backend=InMemoryLongTermMemoryBackend(), embedder=_FakeEmbedder())


@pytest.fixture
def llm():
    """Real OpenAI LLM object — never executed in these tests. Constructed
    only to satisfy Agent's pydantic validation."""
    return OpenAI(
        connection=OpenAIConnection(api_key="test-key"),
        model="gpt-4o",
    )


def _make_agent(llm, *, ltm=None, ltm_config=None) -> Agent:
    kwargs = {"name": "test", "llm": llm, "tools": []}
    if ltm is not None:
        kwargs["long_term_memory"] = ltm
    if ltm_config is not None:
        kwargs["long_term_memory_config"] = ltm_config
    return Agent(**kwargs)


def _input(user_id=None, session_id=None):
    return SimpleNamespace(user_id=user_id, session_id=session_id, input="hi")


# --- LongTermMemoryConfig ---


def test_config_default_includes_remember_and_recall():
    assert LongTermMemoryConfig().tools == ("remember", "recall")


def test_config_can_restrict_to_read_only():
    assert LongTermMemoryConfig(tools=("recall",)).tools == ("recall",)


def test_config_model_dump_emits_plain_strings_not_enums():
    """YAML round-trip relies on tool kinds being dumped as their string values,
    not as enum members (which yaml.safe_dump cannot represent and which would
    round-trip back as the enum *name* — 'REMEMBER' — failing validation)."""
    import yaml

    dumped = LongTermMemoryConfig().model_dump()
    assert dumped == {"tools": ("remember", "recall")}
    assert all(isinstance(t, str) and not hasattr(t, "value") for t in dumped["tools"])
    yaml.safe_dump(dumped)  # must not raise


# --- Agent field declarations ---


def test_agent_has_long_term_memory_fields():
    fields = Agent.model_fields
    assert "long_term_memory" in fields
    assert "long_term_memory_config" in fields
    assert fields["long_term_memory"].default is None


def test_agent_long_term_memory_defaults_to_none(llm):
    agent = _make_agent(llm)
    assert agent.long_term_memory is None
    assert agent.long_term_memory_config.tools == ("remember", "recall")


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


def test_build_respects_config_include(llm, ltm):
    agent = _make_agent(llm, ltm=ltm, ltm_config=LongTermMemoryConfig(tools=("recall",)))
    tools = agent._build_long_term_memory_tools(_input(user_id="u1"))
    assert [t.name for t in tools] == ["recall_facts"]


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


def test_init_components_initializes_ltm_embedder(llm):
    """The embedder is a ConnectionNode whose `text_embedder` client is built
    during `init_components`; without that, the first recall AttributeErrors
    on a None client."""
    init_calls: list = []

    class _RecordingEmbedder(_FakeEmbedder):
        is_postponed_component_init: bool = True

        def init_components(self, connection_manager=None):
            init_calls.append(connection_manager)

    ltm_with_postponed = LongTermMemory(backend=InMemoryLongTermMemoryBackend(), embedder=_RecordingEmbedder())
    agent = _make_agent(llm, ltm=ltm_with_postponed)
    # Node.__init__ already invokes init_components on construction; clear and
    # assert the explicit call also propagates to the embedder.
    init_calls.clear()
    agent.init_components()
    assert len(init_calls) == 1


# --- execute() splice: snapshot/restore self.tools ---


def _patch_run_agent_capture_tools(agent, captured):
    def fake_run(*args, **kwargs):
        captured.extend(agent.tools)
        return "ok"

    return patch.object(agent, "_run_agent", side_effect=fake_run)


def test_execute_attaches_ltm_tools_during_run_and_restores_after(llm, ltm):
    agent = _make_agent(llm, ltm=ltm)
    original_tools = list(agent.tools)
    captured: list = []

    with _patch_run_agent_capture_tools(agent, captured):
        agent.run_sync(input_data={"input": "hi", "user_id": "u1"})

    assert {"remember_fact", "recall_facts"} <= {t.name for t in captured}
    assert agent.tools == original_tools


def test_execute_restores_tools_even_when_run_raises(llm, ltm):
    agent = _make_agent(llm, ltm=ltm)
    original_tools = list(agent.tools)

    with patch.object(agent, "_run_agent", side_effect=RuntimeError("boom")):
        # run_sync wraps exceptions in a failed RunnableResult; check tools after.
        agent.run_sync(input_data={"input": "hi", "user_id": "u1"})

    assert agent.tools == original_tools


def test_execute_does_not_mutate_tools_when_no_user_id(llm, ltm):
    agent = _make_agent(llm, ltm=ltm)
    original_tools = list(agent.tools)
    captured: list = []

    with _patch_run_agent_capture_tools(agent, captured):
        agent.run_sync(input_data={"input": "hi"})

    assert {t.name for t in captured} == {t.name for t in original_tools}
    assert agent.tools == original_tools


def test_execute_does_not_mutate_tools_when_no_long_term_memory(llm):
    agent = _make_agent(llm)
    original_tools = list(agent.tools)
    captured: list = []

    with _patch_run_agent_capture_tools(agent, captured):
        agent.run_sync(input_data={"input": "hi", "user_id": "u1"})

    assert {t.name for t in captured} == {t.name for t in original_tools}
    assert agent.tools == original_tools


def test_execute_preserves_tools_added_mid_run(llm, ltm):
    """Tools appended during execution (e.g. by `_setup_in_memory_file_store_and_tools`)
    must survive LTM cleanup — we remove LTM tools by identity, not by snapshot restore."""
    from dynamiq.nodes.node import Node
    from dynamiq.nodes.types import NodeGroup

    class _FakeFileTool(Node):
        group: ClassVar = NodeGroup.TOOLS
        name: str = "fake_file_tool"

        def execute(self, input_data=None, config=None, **kwargs):
            return {"content": "ok"}

    agent = _make_agent(llm, ltm=ltm)
    original_tools = list(agent.tools)
    injected = _FakeFileTool()

    def fake_run(*args, **kwargs):
        # Simulate `_setup_in_memory_file_store_and_tools` mutating self.tools
        # during the run window — same pattern as the real file-store setup.
        agent.tools = list(agent.tools) + [injected]
        return "ok"

    with patch.object(agent, "_run_agent", side_effect=fake_run):
        agent.run_sync(input_data={"input": "hi", "user_id": "u1"})

    assert injected in agent.tools
    assert all(t.name not in {"remember_fact", "recall_facts"} for t in agent.tools)
    assert agent.tools == original_tools + [injected]


def test_ltm_lock_released_when_post_acquire_mutation_raises(llm, ltm):
    """Anything between lock-acquire and run can raise (list creation, logger
    call). It must still release the lock — otherwise the next LTM-enabled
    execute on this agent would block forever waiting on it."""
    from dynamiq.nodes.agents.base import logger as base_logger

    agent = _make_agent(llm, ltm=ltm)
    real_info = base_logger.info

    def fail_on_ltm_log(msg, *args, **kwargs):
        # Only blow up on the LTM-attach log line so we hit the post-acquire
        # window specifically; let other logger.info calls in execute pass.
        if "long-term memory tools" in str(msg):
            raise RuntimeError("log boom")
        return real_info(msg, *args, **kwargs)

    with patch.object(base_logger, "info", side_effect=fail_on_ltm_log):
        agent.run_sync(input_data={"input": "hi", "user_id": "u1"})

    # Lock must be free now; non-blocking acquire should succeed immediately.
    assert agent._ltm_tools_lock.acquire(blocking=False), "lock was leaked"
    agent._ltm_tools_lock.release()


def test_concurrent_execute_calls_isolate_per_user_ltm_tools(llm, ltm):
    """Two concurrent execute calls with different user_ids must each observe
    only their own LTM tools — the per-agent `_ltm_tools_lock` serialises the
    mutation window so the user-scoped tools never leak across calls."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    agent = _make_agent(llm, ltm=ltm)
    original_tools = list(agent.tools)
    snapshots: dict[str, set[str]] = {}
    snapshots_lock = threading.Lock()

    def fake_run(*args, **kwargs):
        bound_user_ids = {t.user_id for t in agent.tools if hasattr(t, "user_id")}
        assert len(bound_user_ids) == 1, f"cross-user leakage: {bound_user_ids}"
        (uid,) = bound_user_ids
        with snapshots_lock:
            snapshots[uid] = {t.name for t in agent.tools if hasattr(t, "user_id")}
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
    assert agent.tools == original_tools


def test_execute_restores_tools_when_prep_step_raises_before_run(llm, ltm):
    """Regression: prep code between the LTM mutation and the inner try block
    (memory retrieval, file upload, prompt-variable update) used to leak
    appended tools if it raised. The outer try/finally must catch that path."""
    agent = _make_agent(llm, ltm=ltm)
    original_tools = list(agent.tools)

    with patch.object(agent.system_prompt_manager, "update_variables", side_effect=RuntimeError("prep boom")):
        agent.run_sync(input_data={"input": "hi", "user_id": "u1"})

    assert agent.tools == original_tools
