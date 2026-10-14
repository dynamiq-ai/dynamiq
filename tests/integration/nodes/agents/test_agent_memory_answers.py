"""Agent conversation memory: what a run's snapshot stores as the assistant turn.

The assistant turn is the final answer as plain text in every inference mode and save mode,
a canceled run stores only the user input, and a run never deletes a turn another run on the
same session saved while it was running.
"""

import uuid
from unittest.mock import patch

import pytest

from dynamiq import connections
from dynamiq.memory import Memory, MemorySaveMode
from dynamiq.memory.backends import InMemory
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.tools.python import Python
from dynamiq.nodes.types import InferenceMode
from dynamiq.prompts import MessageRole
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus
from dynamiq.types.cancellation import CanceledException

SCOPE = {"user_id": "user-1", "session_id": "session-1"}

TRACE_MARKERS = ("<thought", "<output", "<answer", "Thought:", '"thought"', "provide_final_answer")

ALL_MODES = [
    InferenceMode.DEFAULT,
    InferenceMode.XML,
    InferenceMode.STRUCTURED_OUTPUT,
    InferenceMode.FUNCTION_CALLING,
]


def _final_answer_output(mode: InferenceMode, answer: str) -> dict:
    """The LLM output that makes an agent in ``mode`` finish with ``answer``."""
    if mode == InferenceMode.XML:
        return {
            "content": (
                "<output>\n    <thought>The user greeted me; I can answer directly.</thought>\n"
                f"    <answer>{answer}</answer>\n</output>"
            )
        }
    if mode == InferenceMode.DEFAULT:
        return {"content": f"Thought: The user greeted me; I can answer directly.\nAnswer: {answer}"}
    if mode == InferenceMode.STRUCTURED_OUTPUT:
        return {
            "content": (
                '{"thought": "The user greeted me; I can answer directly.", '
                f'"action": "finish", "action_input": "{answer}"}}'
            )
        }
    return {
        "content": "",
        "tool_calls": [
            {
                "id": f"call_{uuid.uuid4().hex[:6]}",
                "type": "function",
                "function": {
                    "name": "provide_final_answer",
                    "arguments": f'{{"thought": "I can answer directly.", "answer": "{answer}"}}',
                },
            }
        ],
    }


def _tool_call_output(mode: InferenceMode) -> dict:
    """The LLM output that makes an agent in ``mode`` call ``search_tool``."""
    if mode == InferenceMode.XML:
        return {
            "content": (
                "<output>\n    <thought>I should search first.</thought>\n"
                "    <action>search_tool</action>\n"
                '    <action_input>{"q": "X"}</action_input>\n</output>'
            )
        }
    if mode == InferenceMode.FUNCTION_CALLING:
        return {
            "content": "",
            "tool_calls": [
                {
                    "id": "call_tool",
                    "type": "function",
                    "function": {
                        "name": "search_tool",
                        "arguments": '{"thought": "I should search first.", "action_input": {"q": "X"}}',
                    },
                }
            ],
        }
    raise ValueError(mode)


def _result(output: dict) -> RunnableResult:
    return RunnableResult(status=RunnableStatus.SUCCESS, input={}, output=output)


def _agent(mode: InferenceMode, memory: Memory, **kwargs) -> Agent:
    llm = OpenAI(
        name="LLM",
        model="gpt-4o-mini",
        connection=connections.OpenAI(id=str(uuid.uuid4()), api_key="key"),
    )
    tool = Python(name="search_tool", description="Search.", code="def run(input_data): return {'content': 'ok'}")
    return Agent(name="Agent", llm=llm, tools=[tool], inference_mode=mode, memory=memory, **kwargs)


def _run_turn(agent: Agent, question: str, *outputs) -> RunnableResult:
    """Run one turn; each output is an LLM output dict, or a callable returning one."""
    responses = iter(outputs)

    def next_llm_result(*args, **kwargs):
        output = next(responses)
        return _result(output() if callable(output) else output)

    with patch.object(agent, "_run_llm", side_effect=next_llm_result):
        return agent.run(input_data={"input": question, **SCOPE}, config=RunnableConfig())


def _stored(memory: Memory) -> list[tuple]:
    return [(m.role, m.content) for m in memory.get_agent_conversation(filters=SCOPE, limit=100)]


@pytest.mark.parametrize("save_mode", [MemorySaveMode.FULL, MemorySaveMode.INPUT_OUTPUT])
@pytest.mark.parametrize("mode", ALL_MODES)
def test_assistant_turn_is_the_final_answer_not_the_raw_model_output(mode, save_mode):
    memory = Memory(backend=InMemory(), save_mode=save_mode)
    agent = _agent(mode, memory)

    result = _run_turn(agent, "Hi", _final_answer_output(mode, "Hello there!"))
    assert result.status == RunnableStatus.SUCCESS, result.error

    stored = memory.get_agent_conversation(filters=SCOPE)
    assert [(m.role, m.content) for m in stored] == [
        (MessageRole.USER, "Hi"),
        (MessageRole.ASSISTANT, "Hello there!"),
    ]
    assert not stored[-1].tool_calls


@pytest.mark.parametrize("mode", [InferenceMode.XML, InferenceMode.FUNCTION_CALLING])
def test_full_mode_keeps_the_steps_and_ends_with_the_plain_answer(mode):
    memory = Memory(backend=InMemory(), save_mode=MemorySaveMode.FULL)
    agent = _agent(mode, memory)

    result = _run_turn(agent, "Q1", _tool_call_output(mode), _final_answer_output(mode, "A1"))
    assert result.status == RunnableStatus.SUCCESS, result.error

    stored = memory.get_agent_conversation(filters=SCOPE)
    assert stored[0].content == "Q1"
    assert len(stored) > 2, "the tool call and its observation are kept in FULL mode"
    assert stored[-1].role == MessageRole.ASSISTANT
    assert stored[-1].content == "A1"
    assert not stored[-1].tool_calls


@pytest.mark.parametrize("mode", [InferenceMode.XML, InferenceMode.FUNCTION_CALLING])
def test_canceled_run_stores_only_the_user_input(mode):
    memory = Memory(backend=InMemory())
    agent = _agent(mode, memory)
    _run_turn(agent, "Q1", _final_answer_output(mode, "A1"))

    with (
        patch.object(agent, "_run_llm", return_value=_result(_tool_call_output(mode))),
        patch.object(agent, "_execute_tools_and_update_prompt", side_effect=CanceledException()),
    ):
        result = agent.run(input_data={"input": "Q2", **SCOPE}, config=RunnableConfig())
    assert result.status == RunnableStatus.CANCELED

    stored = memory.get_agent_conversation(filters=SCOPE)
    assert [(m.role, m.content) for m in stored] == [
        (MessageRole.USER, "Q1"),
        (MessageRole.ASSISTANT, "A1"),
        (MessageRole.USER, "Q2"),
    ]
    assert not any(marker in m.content for m in stored for marker in TRACE_MARKERS)
    assert not any(m.tool_calls for m in stored)


def test_a_run_keeps_the_turn_another_run_saved_while_it_was_running():
    memory = Memory(backend=InMemory())
    first = _agent(InferenceMode.XML, memory)
    second = _agent(InferenceMode.XML, memory)
    _run_turn(first, "Q0", _final_answer_output(InferenceMode.XML, "A0"))

    def second_run_saves_meanwhile():
        _run_turn(second, "Q2", _final_answer_output(InferenceMode.XML, "A2"))
        return _final_answer_output(InferenceMode.XML, "A1")

    result = _run_turn(first, "Q1", second_run_saves_meanwhile)
    assert result.status == RunnableStatus.SUCCESS, result.error

    contents = [content for _, content in _stored(memory)]
    assert contents[:2] == ["Q0", "A0"]
    assert ["Q2", "A2"] == contents[2:4] or ["Q2", "A2"] == contents[4:6]
    assert sorted(contents[2:]) == ["A1", "A2", "Q1", "Q2"]


def test_without_a_concurrent_run_the_session_is_still_replaced_by_the_snapshot():
    backend = InMemory()
    memory = Memory(backend=backend)
    agent = _agent(InferenceMode.XML, memory)
    _run_turn(agent, "Q0", _final_answer_output(InferenceMode.XML, "A0"))

    with patch.object(InMemory, "delete", autospec=True, side_effect=InMemory.delete) as delete_spy:
        _run_turn(agent, "Q1", _final_answer_output(InferenceMode.XML, "A1"))
    delete_spy.assert_called_once()

    assert [content for _, content in _stored(memory)] == ["Q0", "A0", "Q1", "A1"]
