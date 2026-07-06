"""Unit tests for HumanFeedbackTool streaming behaviour.

Focus: the ``is_browser_takeover`` flag must travel in the streamed event data so a
chat UI can render a browser-takeover interaction instead of a plain text prompt.
"""

from queue import Queue

from dynamiq.callbacks.base import BaseCallbackHandler
from dynamiq.nodes.tools.human_feedback import (
    HFStreamingInputEventMessage,
    HFStreamingInputEventMessageData,
    HumanFeedbackTool,
)
from dynamiq.runnables import RunnableConfig
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
