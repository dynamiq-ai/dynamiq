"""Unit coverage for Agent._unanswered_tool_call_ids().

``_pending_fc_tool_call_ids`` is in-memory scratch state, live only between the LLM
emitting a tool call and the tool returning. An interruption in that window (a HITL
input timeout that checkpoints and resumes) loses it, and the tool result would then
be appended as an unattached ``role: user`` observation, leaving the assistant's
tool_calls unanswered. These tests pin the recovery: the ids are read back out of the
conversation instead.
"""

import uuid

from dynamiq import connections, prompts
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.types import InferenceMode
from dynamiq.prompts import Message, MessageRole


def _make_agent() -> Agent:
    return Agent(
        name="agent",
        llm=OpenAI(
            model="gpt-4o",
            connection=connections.OpenAI(id=str(uuid.uuid4()), api_key="fake-key"),
            prompt=prompts.Prompt(messages=[prompts.Message(role="user", content="{{input}}")]),
        ),
        tools=[],
        inference_mode=InferenceMode.FUNCTION_CALLING,
    )


def _assistant(*ids: str) -> Message:
    return Message(
        role=MessageRole.ASSISTANT,
        content="Calling: " + ", ".join(ids),
        tool_calls=[
            {"id": tc_id, "type": "function", "function": {"name": "some-tool", "arguments": "{}"}} for tc_id in ids
        ],
        static=True,
    )


def _tool_reply(tc_id: str, content: str = "result") -> Message:
    return Message(role=MessageRole.TOOL, content=content, tool_call_id=tc_id, static=True)


def _with_history(*messages: Message) -> Agent:
    agent = _make_agent()
    agent._prompt.messages = [
        Message(role=MessageRole.SYSTEM, content="sys", static=True),
        Message(role=MessageRole.USER, content="do it", static=True),
        *messages,
    ]
    return agent


def test_recovers_id_of_the_outstanding_tool_call():
    """The resume case: an assistant tool_call with no reply yet."""
    agent = _with_history(_assistant("call_1"))
    assert agent._unanswered_tool_call_ids() == ["call_1"]


def test_ignores_tool_calls_that_were_already_answered():
    """A completed exchange leaves nothing outstanding."""
    agent = _with_history(_assistant("call_1"), _tool_reply("call_1"))
    assert agent._unanswered_tool_call_ids() == []


def test_only_the_latest_assistant_turn_is_considered():
    """An older answered call must not be re-answered by a later result."""
    agent = _with_history(
        _assistant("call_1"),
        _tool_reply("call_1"),
        _assistant("call_2"),
    )
    assert agent._unanswered_tool_call_ids() == ["call_2"]


def test_parallel_calls_keep_assistant_order():
    """Order matters: _emit_tool_observations zips these against per-tool results."""
    agent = _with_history(_assistant("call_a", "call_b", "call_c"))
    assert agent._unanswered_tool_call_ids() == ["call_a", "call_b", "call_c"]


def test_partially_answered_batch_returns_only_the_remainder():
    agent = _with_history(_assistant("call_a", "call_b"), _tool_reply("call_a"))
    assert agent._unanswered_tool_call_ids() == ["call_b"]


def test_final_answer_stub_does_not_mask_the_real_tool_call():
    """`provide_final_answer` is acknowledged inline; the real call is still outstanding.

    Mirrors _append_assistant_message, which emits both ids on one assistant message
    and immediately answers only the final-answer stub.
    """
    agent = _with_history(
        _assistant("call_final", "call_real"),
        _tool_reply("call_final", "Acknowledged."),
    )
    assert agent._unanswered_tool_call_ids() == ["call_real"]


def test_plain_assistant_message_yields_nothing():
    """Non-FC output (or an extraction failure) has no ids to recover."""
    agent = _with_history(Message(role=MessageRole.ASSISTANT, content="just text", static=True))
    assert agent._unanswered_tool_call_ids() == []


def test_no_assistant_turn_yields_nothing():
    agent = _with_history()
    assert agent._unanswered_tool_call_ids() == []
