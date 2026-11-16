import json

import pytest

from dynamiq.telemetry import semconv
from dynamiq.telemetry.types import CaptureSettings, PromptRecord

from .helpers import assert_otel_valid, validate_messages

INPUT_MESSAGES = "gen_ai.input.messages"
BASE_SPAN_ID = "dynamiq.prompt.messages_base.span_id"
BASE_COUNT = "dynamiq.prompt.messages_base.count"
SPAN_ID = "00f067aa0ba902b7"

SYSTEM = {"role": "system", "content": "You are an agent with a long system prompt."}
USER = {
    "role": "user",
    "content": [
        {"type": "text", "text": "What is in this picture?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="}},
    ],
}
CALL = {
    "role": "assistant",
    "content": "",
    "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "search", "arguments": '{"query": "picture"}'}}
    ],
}
RESULT = {"role": "tool", "tool_call_id": "call_1", "content": "A long description of the picture."}
ANSWER = {"role": "assistant", "content": "It shows a cat."}


def prompt_attributes(messages, settings, previous=None) -> dict:
    attributes = semconv.prompt_attributes(messages, settings=settings, previous=previous)
    assert_otel_valid(attributes)
    return attributes


def record(messages: list) -> PromptRecord:
    return PromptRecord(span_id=SPAN_ID, messages=messages)


def test_prompt_without_a_previous_one_is_recorded_whole(settings):
    attributes = prompt_attributes([SYSTEM, CALL, RESULT], settings)

    assert set(attributes) == {INPUT_MESSAGES}
    assert [message["role"] for message in validate_messages(attributes[INPUT_MESSAGES], "input")] == [
        "system",
        "assistant",
        "tool",
    ]


def test_prompt_that_continues_the_previous_one_records_what_follows_and_where_the_rest_is(settings):
    attributes = prompt_attributes([SYSTEM, USER, CALL, RESULT], settings, previous=record([SYSTEM, USER]))

    assert attributes[BASE_SPAN_ID] == SPAN_ID
    assert attributes[BASE_COUNT] == 2
    assert type(attributes[BASE_COUNT]) is int
    # What is left is still a list of messages in the convention's shape.
    assert validate_messages(attributes[INPUT_MESSAGES], "input") == [
        {
            "role": "assistant",
            "parts": [{"type": "tool_call", "id": "call_1", "name": "search", "arguments": {"query": "picture"}}],
        },
        {
            "role": "tool",
            "parts": [{"type": "tool_call_response", "id": "call_1", "response": "A long description of the picture."}],
        },
    ]


@pytest.mark.parametrize(
    ("previous", "messages", "count"),
    [
        ([SYSTEM, USER, CALL, RESULT], [SYSTEM, ANSWER], 1),
        ([SYSTEM, USER], [SYSTEM, USER], 2),
        ([SYSTEM, USER, CALL], [SYSTEM, USER], 2),
        ([SYSTEM, USER], [USER, SYSTEM], None),
        ([SYSTEM, USER], [{**SYSTEM, "content": "Another system prompt."}, USER], None),
        ([], [SYSTEM, USER], None),
    ],
    ids=["rewritten-history", "same-prompt", "shorter-prompt", "other-order", "other-first-message", "empty-previous"],
)
def test_shared_start_ends_at_the_first_message_that_differs(settings, previous, messages, count):
    attributes = prompt_attributes(messages, settings, previous=record(previous))

    assert attributes.get(BASE_COUNT) == count
    assert (BASE_SPAN_ID in attributes) == (count is not None)
    assert len(json.loads(attributes[INPUT_MESSAGES])) == len(messages) - (count or 0)


@pytest.mark.parametrize(
    "settings",
    [CaptureSettings(), CaptureSettings(content_max_length=7), CaptureSettings(binary_content_enabled=True)],
    ids=["defaults", "truncated", "binary"],
)
def test_shared_start_of_the_previous_span_and_the_compacted_messages_are_the_whole_prompt(settings):
    # Messages are compared as they are sent and recorded after redaction, binary handling and truncation. The
    # first ``count`` messages the previous span recorded still have to be the ones this prompt starts with.
    previous, messages = [SYSTEM, USER, CALL], [SYSTEM, USER, CALL, RESULT, ANSWER]
    whole = json.loads(prompt_attributes(messages, settings)[INPUT_MESSAGES])
    previous_whole = json.loads(prompt_attributes(previous, settings)[INPUT_MESSAGES])

    compacted = prompt_attributes(messages, settings, previous=record(previous))

    assert compacted[BASE_COUNT] == 3
    assert previous_whole[: compacted[BASE_COUNT]] + json.loads(compacted[INPUT_MESSAGES]) == whole


def test_no_prompt_and_no_pointer_without_content_capture(no_content):
    assert prompt_attributes([SYSTEM, USER, CALL], no_content, previous=record([SYSTEM, USER])) == {}
