"""Cancellation-induced FC orphan: sanitize on memory save.

After a mid-tool cancellation, the agent's `_prompt.messages` would hold an
`assistant(tool_calls=...)` whose matching `role=TOOL` reply was never appended.
`_save_history_to_memory` runs `BaseLLM._sanitize_fc_messages` on
`_prompt.messages` in FC mode before snapshotting — synthetic tool replies are
inserted in place of missing ones, so both the in-flight buffer and the
persisted memory end up balanced.
"""

import uuid
from unittest.mock import patch

from dynamiq import connections, prompts
from dynamiq.memory import Memory
from dynamiq.memory.backends import InMemory
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.tools.python import Python
from dynamiq.nodes.types import InferenceMode
from dynamiq.prompts import MessageRole
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus
from dynamiq.types.cancellation import CanceledException

USER_ID = "user-1"
SESSION_ID = "session-1"


def _slow_tool_factory():
    return Python(
        id=str(uuid.uuid4()),
        name="search_tool",
        description="Pretend search tool",
        code="def run(input_data): return {'content': 'never reached'}",
    )


def _build_agent_with_memory() -> tuple[Agent, Memory]:
    conn = connections.OpenAI(id="fake-conn", api_key="fake-key")
    llm = OpenAI(
        name="TestLLM",
        model="gpt-4o-mini",
        connection=conn,
        prompt=prompts.Prompt(messages=[prompts.Message(role="user", content="{{input}}")]),
    )
    memory = Memory(backend=InMemory())
    agent = Agent(
        name="test-agent",
        llm=llm,
        tools=[_slow_tool_factory()],
        inference_mode=InferenceMode.FUNCTION_CALLING,
        memory=memory,
        max_loops=2,
    )
    return agent, memory


def _llm_returns_tool_call(*args, **kwargs):
    return RunnableResult(
        status=RunnableStatus.SUCCESS,
        input={},
        output={
            "content": "",
            "tool_calls": [
                {
                    "id": "call_A",
                    "type": "function",
                    "function": {
                        "name": "search_tool",
                        "arguments": '{"thought": "searching", "action_input": {"q": "X"}}',
                    },
                }
            ],
        },
    )


def test_sanitization_on_save_clears_orphan_from_memory_and_buffer():
    """After mid-tool cancel, _save_history_to_memory sanitizes _prompt.messages
    in place; both the buffer and the persisted memory are balanced."""
    agent, memory = _build_agent_with_memory()

    with (
        patch.object(agent, "_run_llm", side_effect=_llm_returns_tool_call),
        patch.object(
            agent,
            "_execute_tools_and_update_prompt",
            side_effect=CanceledException(),
        ),
    ):
        result = agent.run(
            input_data={
                "input": "Search the web for X.",
                "user_id": USER_ID,
                "session_id": SESSION_ID,
            },
            config=RunnableConfig(),
        )
        assert result.status == RunnableStatus.CANCELED

    for source_name, messages in (
        ("_prompt.messages", agent._prompt.messages),
        ("memory.get_all()", memory.get_all()),
    ):
        seen_call_ids = {
            tc["id"] for m in messages if m.role == MessageRole.ASSISTANT and m.tool_calls for tc in m.tool_calls
        }
        tool_reply_ids = {m.tool_call_id for m in messages if m.role == MessageRole.TOOL and m.tool_call_id is not None}

        unmatched_calls = seen_call_ids - tool_reply_ids
        assert not unmatched_calls, f"orphan tool_calls in {source_name}: {unmatched_calls}"

        unmatched_replies = tool_reply_ids - seen_call_ids
        assert not unmatched_replies, f"orphan TOOL replies in {source_name}: {unmatched_replies}"


def test_fc_agent_recovers_when_tool_call_missing_required_argument():
    """FC mode: LLM emits a tool_call whose arguments lack the required
    `action_input` field. parse_as_tool_call raises ActionParsingException.
    The agent appends a Correction Instruction and recovers on the next loop."""
    conn = connections.OpenAI(id="fake-conn", api_key="fake-key")
    llm = OpenAI(
        name="TestLLM",
        model="gpt-4o-mini",
        connection=conn,
        prompt=prompts.Prompt(messages=[prompts.Message(role="user", content="{{input}}")]),
    )
    tool = Python(
        id=str(uuid.uuid4()),
        name="search_tool",
        description="Pretend search tool",
        code="def run(input_data): return {'content': 'ok'}",
    )
    agent = Agent(
        name="fc-agent",
        llm=llm,
        tools=[tool],
        inference_mode=InferenceMode.FUNCTION_CALLING,
        max_loops=3,
    )

    def _llm_missing_action_input(*_a, **_kw):
        return RunnableResult(
            status=RunnableStatus.SUCCESS,
            input={},
            output={
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_missing",
                        "type": "function",
                        "function": {"name": "search_tool", "arguments": '{"thought": "x"}'},
                    }
                ],
            },
        )

    def _llm_final(*_a, **_kw):
        return RunnableResult(
            status=RunnableStatus.SUCCESS,
            input={},
            output={
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_done",
                        "type": "function",
                        "function": {
                            "name": "provide_final_answer",
                            "arguments": '{"thought":"done","answer":"OK"}',
                        },
                    }
                ],
            },
        )

    responses = iter([_llm_missing_action_input(), _llm_final()])
    with patch.object(agent, "_run_llm", side_effect=lambda *a, **kw: next(responses)):
        result = agent.run(input_data={"input": "go"}, config=RunnableConfig())
        assert result.status == RunnableStatus.SUCCESS

    recovery = [
        m
        for m in agent._prompt.messages
        if m.role == MessageRole.USER and "Correction Instruction" in (m.content or "")
    ]
    assert recovery, "no recovery instruction added for missing required argument"
    assert "ActionParsingException" in recovery[-1].content
