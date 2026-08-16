"""Unit tests for HumanFeedbackTool.

Covers the streaming behaviour — the ``is_browser_takeover`` flag must travel in the
streamed event data so a chat UI can render a browser-takeover interaction instead of a
plain text prompt — and the console path's behaviour when no console is attached.
"""

import builtins
from queue import Queue

import pytest

from dynamiq.callbacks.base import BaseCallbackHandler
from dynamiq.nodes.tools.human_feedback import (
    HFStreamingInputEventMessage,
    HFStreamingInputEventMessageData,
    HumanFeedbackTool,
)
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types.feedback import FeedbackMethod
from dynamiq.types.streaming import STREAMING_EVENT, StreamingConfig


class _CaptureStreamCallback(BaseCallbackHandler):
    """Records the streaming events a node emits."""

    def __init__(self):
        self.events = []

    def on_node_execute_stream(self, serialized, chunk=None, **kwargs):
        event = kwargs.get("event")
        if event is not None:
            self.events.append(event)


def _preloaded_queue(node_id: str, content: str = "done") -> Queue:
    """A queue holding a single valid streaming-input reply, so the ASK path does not block."""
    queue = Queue()
    queue.put(
        HFStreamingInputEventMessage(
            entity_id=node_id,
            event=STREAMING_EVENT,
            data=HFStreamingInputEventMessageData(content=content),
        ).model_dump_json()
    )
    return queue


def test_ask_stream_event_includes_browser_takeover_flag():
    node_id = "hf-takeover"
    tool = HumanFeedbackTool(
        id=node_id,
        is_browser_takeover=True,
        input_method=FeedbackMethod.STREAM,
        output_method=FeedbackMethod.STREAM,
        streaming=StreamingConfig(enabled=True, input_queue=_preloaded_queue(node_id)),
    )
    capture = _CaptureStreamCallback()

    tool.input_method_streaming(prompt="Take over the browser", config=RunnableConfig(callbacks=[capture]))

    assert capture.events, "expected the ASK prompt to be streamed to the UI"
    assert capture.events[0].data.is_browser_takeover is True


def test_info_stream_event_includes_browser_takeover_flag():
    tool = HumanFeedbackTool(
        id="hf-takeover",
        is_browser_takeover=True,
        output_method=FeedbackMethod.STREAM,
        streaming=StreamingConfig(enabled=True),
    )
    capture = _CaptureStreamCallback()

    tool.output_method_streaming(message="Browser ready for takeover", config=RunnableConfig(callbacks=[capture]))

    assert capture.events, "expected the info message to be streamed to the UI"
    assert capture.events[0].data.is_browser_takeover is True


def test_ask_stream_event_browser_takeover_defaults_to_false():
    node_id = "hf-plain"
    tool = HumanFeedbackTool(
        id=node_id,
        input_method=FeedbackMethod.STREAM,
        output_method=FeedbackMethod.STREAM,
        streaming=StreamingConfig(enabled=True, input_queue=_preloaded_queue(node_id)),
    )
    capture = _CaptureStreamCallback()

    tool.input_method_streaming(prompt="Approve?", config=RunnableConfig(callbacks=[capture]))

    assert capture.events, "expected the ASK prompt to be streamed to the UI"
    assert capture.events[0].data.is_browser_takeover is False


def test_description_reflects_browser_takeover_when_enabled():
    """In takeover mode the agent-facing description must not claim the user can only reply with text."""
    tool = HumanFeedbackTool(is_browser_takeover=True)
    description = tool.description.lower()

    assert "can not perform actions" not in description, "text-only caveat contradicts browser takeover"
    assert "browser" in description, "description should tell the agent the user acts in a live browser"


def test_description_keeps_text_only_caveat_by_default():
    """Without takeover the existing text-only guidance for the agent is preserved."""
    tool = HumanFeedbackTool()

    assert "can not perform actions" in tool.description.lower()


def test_description_is_idempotent_across_rebuilds():
    """Rebuilding from a serialized (already-generated) description must not stack guidance."""
    tool = HumanFeedbackTool(is_browser_takeover=True)
    rebuilt = HumanFeedbackTool(description=tool.description, is_browser_takeover=True)

    assert rebuilt.description == tool.description, "description must be stable across round-trips"
    assert rebuilt.description.lower().count("browser takeover is enabled") == 1
    assert rebuilt.description.count("Message template:") == 1


def test_description_flag_flip_does_not_stack_conflicting_notes():
    """Rebuilding a takeover-flavored description with the flag off must drop the takeover note."""
    takeover = HumanFeedbackTool(is_browser_takeover=True)
    flipped = HumanFeedbackTool(description=takeover.description, is_browser_takeover=False)
    desc = flipped.description.lower()

    assert "browser takeover is enabled" not in desc
    assert "can not perform actions" in desc


class TestConsoleIsUnreadable:
    """An empty answer is a real answer, so a console that could not be read at all must
    not be reported as one — the agent would take it as the human declining to respond.
    """

    @pytest.fixture(autouse=True)
    def no_stdin(self, monkeypatch):
        def _raise(*args, **kwargs):
            raise EOFError("EOF when reading a line")

        monkeypatch.setattr(builtins, "input", _raise)

    def test_asking_fails_instead_of_returning_an_empty_answer(self):
        tool = HumanFeedbackTool(input_method=FeedbackMethod.CONSOLE)

        result = tool.run(input_data={"action": "ask", "input": "Proceed?"})

        assert result.status == RunnableStatus.FAILURE
        assert result.output is None or "" == str(result.output.get("content", ""))

    def test_failure_is_recoverable_so_the_agent_can_react(self):
        tool = HumanFeedbackTool(input_method=FeedbackMethod.CONSOLE)

        result = tool.run(input_data={"action": "ask", "input": "Proceed?"})

        assert result.error.recoverable is True

    def test_message_tells_the_agent_what_to_do_instead(self):
        tool = HumanFeedbackTool(input_method=FeedbackMethod.CONSOLE)

        result = tool.run(input_data={"action": "ask", "input": "Proceed?"})

        assert "console" in result.error.message.lower()

    def test_closed_stdin_behaves_the_same(self, monkeypatch):
        def _raise(*args, **kwargs):
            raise OSError("Bad file descriptor")

        monkeypatch.setattr(builtins, "input", _raise)
        tool = HumanFeedbackTool(input_method=FeedbackMethod.CONSOLE)

        assert tool.run(input_data={"action": "ask", "input": "Proceed?"}).status == RunnableStatus.FAILURE


class TestConsoleAnswersAreUnchanged:
    def test_empty_answer_is_still_a_real_answer(self, monkeypatch):
        monkeypatch.setattr(builtins, "input", lambda *a, **k: "")
        tool = HumanFeedbackTool(input_method=FeedbackMethod.CONSOLE)

        result = tool.run(input_data={"action": "ask", "input": "Proceed?"})

        assert result.status == RunnableStatus.SUCCESS
        assert result.output["content"] == ""

    def test_normal_answer_is_returned(self, monkeypatch):
        monkeypatch.setattr(builtins, "input", lambda *a, **k: "yes, go ahead")
        tool = HumanFeedbackTool(input_method=FeedbackMethod.CONSOLE)

        result = tool.run(input_data={"action": "ask", "input": "Proceed?"})

        assert result.status == RunnableStatus.SUCCESS
        assert result.output["content"] == "yes, go ahead"
