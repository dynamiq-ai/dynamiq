import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.memory.notes import NotesConfig
from dynamiq.memory.notes.backends import InMemoryNotesBackend
from dynamiq.nodes.agents.agent import Agent as ReactAgent
from dynamiq.nodes.agents.agent import ReactStep
from dynamiq.nodes.agents.base import NOTES_INDEX_UNAVAILABLE, Agent, _run_notes_index
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.types import InferenceMode

NOTES_SECTION = "YOUR SAVED NOTES"
NOTES_PROSE = "## Notes"
TOOL_NAMES = {"write_note", "read_note", "delete_note"}


@pytest.fixture
def llm():
    """Real OpenAI LLM object — never executed in these tests. Constructed only to
    satisfy Agent's pydantic validation."""
    return OpenAI(connection=OpenAIConnection(api_key="test-key"), model="gpt-4o")


def _notes_config(*, enabled=True, backend=None, **kwargs) -> NotesConfig:
    return NotesConfig(backend=backend or InMemoryNotesBackend(), enabled=enabled, **kwargs)


def _make_agent(llm, *, notes=None, react=False, **kwargs) -> Agent:
    cls = ReactAgent if react else Agent
    agent_kwargs = {"name": "test", "llm": llm, "tools": []} | kwargs
    if notes is not None:
        agent_kwargs["notes"] = notes
    return cls(**agent_kwargs)


def _input(user_id=None, session_id=None):
    return SimpleNamespace(user_id=user_id, session_id=session_id, input="hi")


def _seed(backend, user_id="u1"):
    backend.write(
        user_id=user_id,
        title="Deploy runbook",
        description="ship steps and rollback",
        content="THE-RUNBOOK-BODY",
    )
    backend.write(
        user_id=user_id,
        title="API conventions",
        description="error envelope rules",
        content="THE-CONVENTIONS-BODY",
    )
    return backend


def _patch_run_agent_capture_prompt(agent, captured):
    """Build the real system prompt mid-run and capture it, without calling an LLM."""

    def fake_run(input_message, history_messages=None, *args, **kwargs):
        agent._setup_prompt_and_stop_sequences(input_message, history_messages)
        captured.append(agent._prompt.messages[0].content)
        return "ok"

    return patch.object(agent, "_run_agent", side_effect=fake_run)


def _patch_run_agent_capture_runtime_tools(agent, captured):
    def fake_run(*args, **kwargs):
        captured.append(set(agent.tool_by_names.keys()))
        return "ok"

    return patch.object(agent, "_run_agent", side_effect=fake_run)


# --- NotesConfig / agent field ---------------------------------------------------------


def test_config_defaults_to_enabled():
    assert _notes_config().enabled is True


def test_agent_has_notes_field():
    assert "notes" in Agent.model_fields
    assert Agent.model_fields["notes"].default is None


def test_agent_notes_defaults_to_none(llm):
    assert _make_agent(llm).notes is None


def test_to_dict_serializes_notes_without_the_live_backend(llm):
    agent = _make_agent(llm, notes=_notes_config())

    data = agent.to_dict()

    assert data["notes"]["enabled"] is True
    assert data["notes"]["type"].endswith("NotesConfig")
    assert isinstance(data["notes"]["backend"], dict)
    assert data["notes"]["backend"]["type"] == "dynamiq.memory.notes.backends.InMemoryNotesBackend"


def test_to_dict_notes_is_none_when_unset(llm):
    assert _make_agent(llm).to_dict()["notes"] is None


# --- per-run construction --------------------------------------------------------------


def test_build_returns_tools_and_index_when_enabled(llm):
    agent = _make_agent(llm, notes=_notes_config(backend=_seed(InMemoryNotesBackend())))

    tools, index = agent._build_notes_runtime(_input(user_id="u1"))

    assert {tool.name for tool in tools} == TOOL_NAMES
    assert "Deploy runbook — ship steps and rollback" in index


def test_build_bakes_user_id_into_each_tool(llm):
    agent = _make_agent(llm, notes=_notes_config())

    tools, _ = agent._build_notes_runtime(_input(user_id="u1"))

    assert all(tool.user_id == "u1" for tool in tools)


def test_build_sets_is_optimized_for_agents_on_each_tool(llm):
    agent = _make_agent(llm, notes=_notes_config())

    tools, _ = agent._build_notes_runtime(_input(user_id="u1"))

    assert all(tool.is_optimized_for_agents for tool in tools)


def test_build_raises_when_notes_enabled_but_no_user_id(llm):
    agent = _make_agent(llm, notes=_notes_config())

    with pytest.raises(ValueError, match="user_id"):
        agent._build_notes_runtime(_input(session_id="s1"))


@pytest.mark.parametrize("notes", [None, "disabled"])
def test_build_returns_nothing_when_notes_off(llm, notes):
    config = None if notes is None else _notes_config(enabled=False)
    agent = _make_agent(llm, notes=config)

    assert agent._build_notes_runtime(_input(user_id="u1")) == ([], "")


def test_build_degrades_when_the_backend_fails(llm):
    backend = InMemoryNotesBackend()
    agent = _make_agent(llm, notes=_notes_config(backend=backend))

    with patch.object(InMemoryNotesBackend, "index", side_effect=RuntimeError("db down")):
        tools, index = agent._build_notes_runtime(_input(user_id="u1"))

    assert index == NOTES_INDEX_UNAVAILABLE
    assert {tool.name for tool in tools} == TOOL_NAMES


def test_index_is_truncated_visibly_when_over_max_chars(llm):
    backend = InMemoryNotesBackend()
    for i in range(20):
        backend.write(user_id="u1", title=f"note-{i:02d}", description="d" * 40, content="c")
    agent = _make_agent(llm, notes=_notes_config(backend=backend, max_index_chars=200))

    _, index = agent._build_notes_runtime(_input(user_id="u1"))

    assert len(index) <= 250
    assert "index truncated" in index


# --- prompt injection ------------------------------------------------------------------


def test_index_appears_in_the_system_prompt(llm):
    agent = _make_agent(llm, notes=_notes_config(backend=_seed(InMemoryNotesBackend())), react=True)
    captured: list[str] = []

    with _patch_run_agent_capture_prompt(agent, captured):
        agent.run_sync(input_data={"input": "hi", "user_id": "u1"})

    prompt = captured[0]
    assert NOTES_SECTION in prompt
    assert "- Deploy runbook — ship steps and rollback" in prompt
    assert "- API conventions — error envelope rules" in prompt
    assert NOTES_PROSE in prompt


def test_note_bodies_never_reach_the_system_prompt(llm):
    agent = _make_agent(llm, notes=_notes_config(backend=_seed(InMemoryNotesBackend())), react=True)
    captured: list[str] = []

    with _patch_run_agent_capture_prompt(agent, captured):
        agent.run_sync(input_data={"input": "hi", "user_id": "u1"})

    assert "THE-RUNBOOK-BODY" not in captured[0]
    assert "THE-CONVENTIONS-BODY" not in captured[0]


def test_empty_index_omits_the_section_but_keeps_the_prose(llm):
    agent = _make_agent(llm, notes=_notes_config(), react=True)
    captured: list[str] = []

    with _patch_run_agent_capture_prompt(agent, captured):
        agent.run_sync(input_data={"input": "hi", "user_id": "u1"})

    assert "(no notes yet)" in captured[0]
    assert NOTES_PROSE in captured[0]


def test_disabled_notes_render_neither_section_nor_prose(llm):
    agent = _make_agent(llm, notes=_notes_config(enabled=False), react=True)
    captured: list[str] = []

    with _patch_run_agent_capture_prompt(agent, captured):
        agent.run_sync(input_data={"input": "hi", "user_id": "u1"})

    assert NOTES_SECTION not in captured[0]
    assert NOTES_PROSE not in captured[0]
    assert agent.system_prompt_manager._prompt_blocks.get("notes", "") == ""


def test_note_titles_are_never_rendered_as_jinja(llm):
    """The `notes` block is a placeholder and the index arrives as a substituted value, so
    an LLM-authored title containing template syntax must survive verbatim."""
    backend = InMemoryNotesBackend()
    hostile = "{{ user_id }} {% raw %}x{% endraw %}"
    backend.write(user_id="u1", title=hostile, description="hostile title", content="body")
    agent = _make_agent(llm, notes=_notes_config(backend=backend), react=True)
    captured: list[str] = []

    with _patch_run_agent_capture_prompt(agent, captured):
        agent.run_sync(input_data={"input": "hi", "user_id": "u1"})

    assert hostile in captured[0]


def test_an_input_key_cannot_spoof_the_notes_index(llm):
    agent = _make_agent(llm, notes=_notes_config(backend=_seed(InMemoryNotesBackend())), react=True)
    captured: list[str] = []

    with _patch_run_agent_capture_prompt(agent, captured):
        agent.run_sync(input_data={"input": "hi", "user_id": "u1", "notes_index": "FAKE-INDEX"})

    assert "FAKE-INDEX" not in captured[0]
    assert "Deploy runbook" in captured[0]


def test_xml_prompt_includes_tool_blocks_when_only_notes_configured(llm):
    agent = _make_agent(llm, notes=_notes_config(), react=True, inference_mode=InferenceMode.XML)

    assert "{{ tool_description }}" in agent.system_prompt_manager._prompt_blocks["tools"]


def test_xml_prompt_omits_tool_blocks_when_notes_disabled(llm):
    agent = _make_agent(llm, notes=_notes_config(enabled=False), react=True, inference_mode=InferenceMode.XML)

    assert agent.system_prompt_manager._prompt_blocks["tools"] == ""


# --- isolation -------------------------------------------------------------------------


def test_notes_tools_are_visible_during_the_run_only(llm):
    agent = _make_agent(llm, notes=_notes_config())
    original_tools = list(agent.tools)
    captured: list[set[str]] = []

    with _patch_run_agent_capture_runtime_tools(agent, captured):
        agent.run_sync(input_data={"input": "hi", "user_id": "u1"})

    assert TOOL_NAMES <= captured[0]
    assert agent.tools == original_tools
    assert TOOL_NAMES.isdisjoint(agent.tool_by_names.keys())


def test_overlay_is_cleared_even_when_the_run_raises(llm):
    agent = _make_agent(llm, notes=_notes_config())

    with patch.object(agent, "_run_agent", side_effect=RuntimeError("boom")):
        agent.run_sync(input_data={"input": "hi", "user_id": "u1"})

    assert TOOL_NAMES.isdisjoint(agent.tool_by_names.keys())


def test_execute_surfaces_failure_when_notes_enabled_but_no_user_id(llm):
    agent = _make_agent(llm, notes=_notes_config())

    result = agent.run_sync(input_data={"input": "hi", "session_id": "s1"})

    assert result.status.value == "failure"


def _two_user_backend() -> InMemoryNotesBackend:
    """Titles chosen not to collide with the examples inside NOTES_TOOLS_INSTRUCTIONS,
    so an assertion can distinguish the injected index from the prompt's own prose."""
    backend = InMemoryNotesBackend()
    backend.write(user_id="u1", title="Zebra alpha", description="belongs to u1", content="body")
    backend.write(user_id="u2", title="Quokka beta", description="belongs to u2", content="body")
    return backend


def test_sequential_runs_for_two_users_do_not_leak(llm):
    agent = _make_agent(llm, notes=_notes_config(backend=_two_user_backend()), react=True)
    captured: list[str] = []

    with _patch_run_agent_capture_prompt(agent, captured):
        agent.run_sync(input_data={"input": "hi", "user_id": "u1"})
        agent.run_sync(input_data={"input": "hi", "user_id": "u2"})

    first, second = captured
    assert "Zebra alpha" in first and "Quokka beta" not in first
    assert "Quokka beta" in second and "Zebra alpha" not in second


def test_concurrent_runs_isolate_per_user_indexes(llm):
    """Asserts on the ContextVar rather than the rendered prompt: `_prompt` is instance
    state shared by both runs, so only the ContextVar can carry per-run isolation."""
    agent = _make_agent(llm, notes=_notes_config(backend=_two_user_backend()), react=True)
    barrier = threading.Barrier(2)
    seen: dict[str, str] = {}
    lock = threading.Lock()

    def fake_run(*args, **kwargs):
        barrier.wait()
        index = _run_notes_index.get()
        with lock:
            seen["u1" if "Zebra alpha" in index else "u2"] = index
        return "ok"

    def run(user_id):
        agent.run_sync(input_data={"input": "hi", "user_id": user_id})

    with patch.object(agent, "_run_agent", side_effect=fake_run):
        threads = [threading.Thread(target=run, args=(uid,)) for uid in ("u1", "u2")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    assert set(seen) == {"u1", "u2"}
    assert "Quokka beta" not in seen["u1"]
    assert "Zebra alpha" not in seen["u2"]


def test_react_loop_dispatches_a_notes_tool_when_the_agent_has_no_static_tools(llm):
    """The whole feature depends on this: notes tools live only in the per-run overlay, so
    an agent built with `tools=[]` must still dispatch them. `_execute_tools_and_update_prompt`
    previously gated on `self.tools`, which is empty here — the action was parsed, never
    executed, and the agent looped to max_loops writing nothing.
    """
    backend = InMemoryNotesBackend()
    agent = _make_agent(llm, notes=_notes_config(backend=backend), react=True)
    steps = [
        ReactStep(
            kind="tool_call",
            thought="saving",
            action="write_note",
            action_input={"title": "Runbook", "description": "ship steps", "content": "1. merge"},
        ),
        ReactStep(kind="final_answer", thought="done", final_answer="saved"),
    ]

    with patch.object(agent, "_run_react_llm_step", side_effect=steps):
        result = agent.run_sync(input_data={"input": "save this", "user_id": "u1"})

    assert result.status.value == "success"
    stored = backend.list_all(user_id="u1")
    assert [note.title for note in stored] == ["Runbook"]
    # The tool result must reach the model as an Observation, not vanish.
    observations = [m.content for m in agent._prompt.messages if m.content.startswith("Observation:")]
    assert any("Created note 'Runbook'" in o for o in observations)


def test_sub_agent_without_notes_does_not_inherit_the_overlay(llm):
    parent = _make_agent(llm, notes=_notes_config())
    sub_agent = _make_agent(llm, name="sub")
    sub_captured: list[set[str]] = []

    def parent_run(*args, **kwargs):
        with _patch_run_agent_capture_runtime_tools(sub_agent, sub_captured):
            sub_agent.run_sync(input_data={"input": "hi", "user_id": "u1"})
        return "ok"

    with patch.object(parent, "_run_agent", side_effect=parent_run):
        parent.run_sync(input_data={"input": "hi", "user_id": "u1"})

    assert TOOL_NAMES.isdisjoint(sub_captured[0])
