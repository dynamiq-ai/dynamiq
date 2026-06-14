"""Cancellation-induced FC orphan: sanitize on memory save.

After a mid-tool cancellation, the agent's `_prompt.messages` would hold an
`assistant(tool_calls=...)` whose matching `role=TOOL` reply was never appended.
`_save_history_to_memory` runs `BaseLLM._sanitize_fc_messages` on a local copy
of `_prompt.messages` in FC mode before snapshotting, so the persisted memory
is replay-safe (synthetic tool replies are inserted in place of missing ones)
while `_prompt.messages` itself stays as the canonical in-flight conversation
— the dispatch-time sanitizer repairs orphans again on the next LLM call.
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


def test_sanitization_on_save_clears_orphan_from_memory():
    """After mid-tool cancel, _save_history_to_memory sanitizes a local copy
    before snapshotting: persisted memory is balanced (replay-safe), while
    _prompt.messages stays canonical and may still carry the orphan tool_call
    — the dispatch-time sanitizer repairs it on the next LLM call."""
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

    memory_messages = memory.get_all()
    seen_call_ids = {
        tc["id"] for m in memory_messages if m.role == MessageRole.ASSISTANT and m.tool_calls for tc in m.tool_calls
    }
    tool_reply_ids = {
        m.tool_call_id for m in memory_messages if m.role == MessageRole.TOOL and m.tool_call_id is not None
    }
    assert not (seen_call_ids - tool_reply_ids), "orphan tool_calls leaked into persisted memory"
    assert not (tool_reply_ids - seen_call_ids), "orphan TOOL replies leaked into persisted memory"

    buffer_call_ids = {
        tc["id"]
        for m in agent._prompt.messages
        if m.role == MessageRole.ASSISTANT and m.tool_calls
        for tc in m.tool_calls
    }
    buffer_reply_ids = {
        m.tool_call_id for m in agent._prompt.messages if m.role == MessageRole.TOOL and m.tool_call_id is not None
    }
    assert (
        buffer_call_ids - buffer_reply_ids
    ), "expected _prompt.messages to retain the cancelled orphan tool_call (canonical state policy)"


def test_fc_agent_recovers_when_tool_call_arguments_are_malformed_json():
    """FC mode: LLM emits a tool_call whose ``arguments`` string is not valid JSON.

    ``FunctionCall.parse_arguments`` raises ValueError → wrapped as
    ActionParsingException → the agent appends a "Tool call failed" recovery reply.
    Under the canonical-state FC policy the malformed assistant ``tool_call`` is
    kept in history, so the correction is delivered as a TOOL message replying
    to that ``tool_call_id`` (a USER message would orphan the tool_call), and the
    agent recovers on the next loop.
    """
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

    def _llm_malformed_arguments(*_a, **_kw):
        return RunnableResult(
            status=RunnableStatus.SUCCESS,
            input={},
            output={
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_bad",
                        "type": "function",
                        "function": {
                            "name": "search_tool",
                            # Not valid JSON — triggers parse_arguments validator failure.
                            "arguments": "not valid json {",
                        },
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

    responses = iter([_llm_malformed_arguments(), _llm_final()])
    with patch.object(agent, "_run_llm", side_effect=lambda *a, **kw: next(responses)):
        result = agent.run(input_data={"input": "go"}, config=RunnableConfig())
        assert result.status == RunnableStatus.SUCCESS

    recovery = [
        m for m in agent._prompt.messages if m.role == MessageRole.TOOL and "Tool call failed" in (m.content or "")
    ]
    assert recovery, "no recovery instruction added for malformed tool_call arguments"
    assert "ActionParsingException" in recovery[-1].content
    # Canonical-state policy: the correction replies to the malformed tool_call's id
    # so no orphan tool_call is left in history.
    assert recovery[-1].tool_call_id == "call_bad"
