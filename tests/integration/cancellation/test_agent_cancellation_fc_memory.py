"""Regression test for the cancellation-induced FC orphan-in-memory bug.

With the deferred-append refactor, `_run_agent` no longer appends
`assistant(tool_calls=...)` to history before tool execution. The payload is
stashed in `_pending_assistant_payload` and flushed only inside
`_emit_tool_observations` — i.e., immediately before the matching
`role=TOOL` replies are appended.

Cancellation during tool execution exits via exception before the flush, so
history never holds an orphan. The cancellation handler in `BaseAgent.execute`
then persists the (balanced) history to memory.

This test exercises the end-to-end path: a real Agent in FC mode with
memory, a mocked LLM that returns a tool_call, a tool that raises
CanceledException mid-execution. Asserts memory contains no
`assistant.tool_calls` with no matching `role=TOOL` reply.
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
    """A simple Python tool the agent can decide to call."""
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
    """Mock an LLM response that emits a single search_tool call."""
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


def test_cancellation_during_tool_execution_leaves_no_orphan_in_memory():
    """Mid-tool cancellation must not persist an orphan `assistant(tool_calls)`
    into memory. With the deferred-append flow, the assistant message is only
    appended right before tool replies — so a cancellation that interrupts
    tool execution leaves history balanced and memory clean."""
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

    persisted = memory.get_all()

    # No assistant message with `tool_calls` should appear without its matching reply.
    seen_call_ids = {
        tc["id"]
        for m in persisted
        if m.role == MessageRole.ASSISTANT and m.tool_calls
        for tc in m.tool_calls
    }
    tool_reply_ids = {
        m.tool_call_id for m in persisted if m.role == MessageRole.TOOL and m.tool_call_id is not None
    }

    unmatched_calls = seen_call_ids - tool_reply_ids
    assert not unmatched_calls, (
        f"orphan tool_call_ids persisted into memory after cancellation: {unmatched_calls}"
    )

    unmatched_replies = tool_reply_ids - seen_call_ids
    assert not unmatched_replies, (
        f"orphan TOOL replies persisted into memory after cancellation: {unmatched_replies}"
    )


def test_fc_parse_error_recovery_prompt_includes_tool_calls():
    """When FC parsing fails, the recovery instruction must show the LLM what
    it tried. Since FC output lives in `tool_calls` (not `content`), the
    recovery handler serializes `llm_result.output["tool_calls"]` as JSON text
    and feeds it as `llm_generated_output` to `_append_recovery_instruction`.
    The LLM sees its previous attempt in the next iteration's prompt."""
    agent, _ = _build_agent_with_memory()

    bad_response = RunnableResult(
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
                        "arguments": "{not valid json",  # malformed → ActionParsingException
                    },
                }
            ],
        },
    )
    final_response = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input={},
        output={
            "content": "",
            "tool_calls": [
                {
                    "id": "call_final",
                    "type": "function",
                    "function": {
                        "name": "provide_final_answer",
                        "arguments": '{"thought": "done", "answer": "all good", "output_files": ""}',
                    },
                }
            ],
        },
    )

    with patch.object(agent, "_run_llm", side_effect=[bad_response, final_response]):
        result = agent.run(
            input_data={"input": "Search the web for X.", "user_id": USER_ID, "session_id": SESSION_ID},
            config=RunnableConfig(),
        )

    assert result.status == RunnableStatus.SUCCESS

    # Locate the recovery-instruction assistant message added after parse failure.
    recovery = next(
        (
            m
            for m in agent._prompt.messages
            if m.role == MessageRole.ASSISTANT
            and isinstance(m.content, str)
            and m.content.startswith("Previous response:")
        ),
        None,
    )
    assert recovery is not None, "no recovery-instruction assistant message in history"

    # The recovery prompt must surface the structured tool_calls payload as text
    # so the LLM can see what it tried.
    assert "call_bad" in recovery.content, "tool_call id missing from recovery context"
    assert "search_tool" in recovery.content, "tool name missing from recovery context"
    assert "{not valid json" in recovery.content, "tool_call arguments missing from recovery context"
