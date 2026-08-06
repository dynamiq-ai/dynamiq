"""Unit tests for HumanFeedbackTool streaming behaviour.

Focus: the ``is_browser_takeover`` flag must travel in the streamed event data so a
chat UI can render a browser-takeover interaction instead of a plain text prompt.
"""

from queue import Queue

import pytest
from pydantic import ValidationError

from dynamiq.callbacks.base import BaseCallbackHandler
from dynamiq.nodes.tools.human_feedback import (
    Answer,
    HFStreamingInputEventMessage,
    HFStreamingInputEventMessageData,
    HumanFeedbackAction,
    HumanFeedbackInputSchema,
    HumanFeedbackTool,
    Question,
    QuestionOption,
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


def test_question_requires_two_to_four_options_when_given():
    with pytest.raises(ValidationError):
        Question(question="Pick one", options=[QuestionOption(label="only one")])

    Question(question="Pick one", options=[QuestionOption(label="a"), QuestionOption(label="b")])


def test_input_schema_requires_one_to_four_questions_when_given():
    with pytest.raises(ValidationError):
        HumanFeedbackInputSchema(action=HumanFeedbackAction.ASK, questions=[])

    HumanFeedbackInputSchema(action=HumanFeedbackAction.ASK, questions=[Question(question="Any concerns?")])


def test_input_method_streaming_sends_questions_and_mints_request_id():
    node_id = "hf-questions"
    tool = HumanFeedbackTool(
        id=node_id,
        input_method=FeedbackMethod.STREAM,
        output_method=FeedbackMethod.STREAM,
        streaming=StreamingConfig(enabled=True, input_queue=_preloaded_queue(node_id)),
    )
    capture = _CaptureStreamCallback()
    questions = [
        Question(
            question="Which scope?",
            options=[QuestionOption(label="Q1 only"), QuestionOption(label="Full year")],
        )
    ]

    tool.input_method_streaming(prompt="pick one", config=RunnableConfig(callbacks=[capture]), questions=questions)

    sent = capture.events[0].data
    assert sent.questions[0].question == "Which scope?"
    assert sent.questions[0].id == "0", "unset question id should default to its batch index"
    assert sent.request_id, "every ask, structured or not, must mint a request_id"


def test_get_input_streaming_event_discards_stale_request_id():
    """A reply left over from an earlier question must not satisfy the current one (stale-answer replay)."""
    node_id = "hf-multi-round"
    tool = HumanFeedbackTool(id=node_id, input_method=FeedbackMethod.STREAM, output_method=FeedbackMethod.STREAM)
    queue = Queue()
    tool.streaming = StreamingConfig(enabled=True, input_queue=queue, timeout=5.0)
    queue.put(
        HFStreamingInputEventMessage(
            entity_id=node_id,
            event=STREAMING_EVENT,
            data=HFStreamingInputEventMessageData(content="stale answer", request_id="round-1"),
        ).model_dump_json()
    )
    queue.put(
        HFStreamingInputEventMessage(
            entity_id=node_id,
            event=STREAMING_EVENT,
            data=HFStreamingInputEventMessageData(content="fresh answer", request_id="round-2"),
        ).model_dump_json()
    )

    result = tool.get_input_streaming_event(
        event_msg_type=HFStreamingInputEventMessage,
        event=STREAMING_EVENT,
        config=RunnableConfig(),
        request_id="round-2",
    )

    assert result.data.content == "fresh answer"


def test_get_input_streaming_event_accepts_reply_without_request_id():
    """Backward compat: a reply from a sender that doesn't echo request_id is still accepted."""
    node_id = "hf-legacy-client"
    tool = HumanFeedbackTool(id=node_id, input_method=FeedbackMethod.STREAM, output_method=FeedbackMethod.STREAM)
    queue = Queue()
    tool.streaming = StreamingConfig(enabled=True, input_queue=queue, timeout=5.0)
    queue.put(
        HFStreamingInputEventMessage(
            entity_id=node_id, event=STREAMING_EVENT, data=HFStreamingInputEventMessageData(content="plain reply")
        ).model_dump_json()
    )

    result = tool.get_input_streaming_event(
        event_msg_type=HFStreamingInputEventMessage,
        event=STREAMING_EVENT,
        config=RunnableConfig(),
        request_id="round-2",
    )

    assert result.data.content == "plain reply"


def test_execute_ask_formats_structured_answers_for_llm_observation():
    node_id = "hf-answered"
    queue = Queue()
    queue.put(
        HFStreamingInputEventMessage(
            entity_id=node_id,
            event=STREAMING_EVENT,
            data=HFStreamingInputEventMessageData(answers=[Answer(question_id="0", selected=["Full year"])]),
        ).model_dump_json()
    )
    tool = HumanFeedbackTool(
        id=node_id,
        input_method=FeedbackMethod.STREAM,
        output_method=FeedbackMethod.STREAM,
        streaming=StreamingConfig(enabled=True, input_queue=queue),
    )
    questions = [
        Question(
            question="Which scope?",
            options=[QuestionOption(label="Q1 only"), QuestionOption(label="Full year")],
        )
    ]

    result = tool.execute(
        HumanFeedbackInputSchema(action=HumanFeedbackAction.ASK, questions=questions), config=RunnableConfig()
    )

    assert result["answers"] == [{"question_id": "0", "selected": ["Full year"], "custom_text": None}]
    assert "Which scope?" in result["content"]
    assert "Full year" in result["content"]


def test_execute_ask_plain_reply_has_no_answers_key():
    """Unstructured ask/reply keeps today's output shape - no 'answers' key at all."""
    node_id = "hf-plain-reply"
    tool = HumanFeedbackTool(
        id=node_id,
        input_method=FeedbackMethod.STREAM,
        output_method=FeedbackMethod.STREAM,
        streaming=StreamingConfig(enabled=True, input_queue=_preloaded_queue(node_id, content="42")),
    )

    result = tool.execute(HumanFeedbackInputSchema(action=HumanFeedbackAction.ASK, input="?"), config=RunnableConfig())

    assert result == {"content": "42"}


def test_execute_ask_ignores_questions_in_browser_takeover_mode():
    """Structured questions don't apply when the user is acting directly in a live browser."""
    node_id = "hf-takeover-questions"
    tool = HumanFeedbackTool(
        id=node_id,
        is_browser_takeover=True,
        input_method=FeedbackMethod.STREAM,
        output_method=FeedbackMethod.STREAM,
        streaming=StreamingConfig(enabled=True, input_queue=_preloaded_queue(node_id)),
    )
    capture = _CaptureStreamCallback()
    questions = [Question(question="Ignored?", options=[QuestionOption(label="a"), QuestionOption(label="b")])]

    tool.execute(
        HumanFeedbackInputSchema(action=HumanFeedbackAction.ASK, questions=questions),
        config=RunnableConfig(callbacks=[capture]),
    )

    assert capture.events[0].data.questions is None
