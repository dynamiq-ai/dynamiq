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
    InputMethodCallable,
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


def test_input_method_streaming_sends_questions():
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


def test_question_coerces_bare_string_options_to_labels():
    question = Question(question="Pick one", options=["Yes", {"label": "No", "description": "Skip it"}])

    assert question.options == [QuestionOption(label="Yes"), QuestionOption(label="No", description="Skip it")]


def test_input_schema_accepts_bare_string_options_from_tool_call():
    schema = HumanFeedbackInputSchema.model_validate(
        {"action": "ask", "questions": [{"question": "Which scope?", "options": ["Q1 only", "Full year"]}]}
    )

    assert [option.label for option in schema.questions[0].options] == ["Q1 only", "Full year"]


def test_description_shows_options_as_label_objects():
    tool = HumanFeedbackTool()

    assert '"options": [{"label": ' in tool.description
    assert "not bare strings" in tool.description


def _scope_questions() -> list[Question]:
    return [
        Question(
            question="Which scope?",
            options=[QuestionOption(label="Q1 only"), QuestionOption(label="Full year")],
        )
    ]


def test_execute_ask_streams_authored_prompt_and_separate_text_fallback():
    node_id = "hf-fallback"
    tool = HumanFeedbackTool(
        id=node_id,
        input_method=FeedbackMethod.STREAM,
        output_method=FeedbackMethod.STREAM,
        streaming=StreamingConfig(enabled=True, input_queue=_preloaded_queue(node_id)),
    )
    capture = _CaptureStreamCallback()

    tool.execute(
        HumanFeedbackInputSchema(action=HumanFeedbackAction.ASK, input="Before I start:", questions=_scope_questions()),
        config=RunnableConfig(callbacks=[capture]),
    )

    sent = capture.events[0].data
    assert sent.prompt == "Before I start:"
    assert (
        sent.prompt_text_fallback
        == "Before I start:\n1. Which scope?\n   - Q1 only\n   - Full year\n   - Other (free text)"
    )


def test_execute_ask_questions_only_streams_empty_prompt_with_fallback():
    node_id = "hf-questions-only"
    tool = HumanFeedbackTool(
        id=node_id,
        input_method=FeedbackMethod.STREAM,
        output_method=FeedbackMethod.STREAM,
        streaming=StreamingConfig(enabled=True, input_queue=_preloaded_queue(node_id)),
    )
    capture = _CaptureStreamCallback()

    tool.execute(
        HumanFeedbackInputSchema(action=HumanFeedbackAction.ASK, questions=_scope_questions()),
        config=RunnableConfig(callbacks=[capture]),
    )

    sent = capture.events[0].data
    assert sent.prompt == ""
    assert sent.prompt_text_fallback.startswith("1. Which scope?")


def test_execute_ask_without_questions_has_no_text_fallback():
    node_id = "hf-no-fallback"
    tool = HumanFeedbackTool(
        id=node_id,
        input_method=FeedbackMethod.STREAM,
        output_method=FeedbackMethod.STREAM,
        streaming=StreamingConfig(enabled=True, input_queue=_preloaded_queue(node_id)),
    )
    capture = _CaptureStreamCallback()

    tool.execute(
        HumanFeedbackInputSchema(action=HumanFeedbackAction.ASK, input="Approve?"),
        config=RunnableConfig(callbacks=[capture]),
    )

    sent = capture.events[0].data
    assert sent.prompt == "Approve?"
    assert sent.prompt_text_fallback is None


def test_execute_ask_text_only_input_method_still_gets_rendered_questions():
    class _RecordingInput(InputMethodCallable):
        def __init__(self):
            self.prompts = []

        def get_input(self, prompt: str, **kwargs) -> str:
            self.prompts.append(prompt)
            return "Full year"

    recorder = _RecordingInput()
    tool = HumanFeedbackTool(input_method=recorder)

    tool.execute(
        HumanFeedbackInputSchema(action=HumanFeedbackAction.ASK, input="Before I start:", questions=_scope_questions()),
        config=RunnableConfig(),
    )

    assert recorder.prompts == [
        "Before I start:\n1. Which scope?\n   - Q1 only\n   - Full year\n   - Other (free text)"
    ]
