"""Prompt rendering and token counting must be fast *and* unchanged.

Both optimizations here are caches, so the tests that matter are the ones proving the
cached answer equals the uncached one -- including after mutation.
"""
import logging

import pytest
from litellm import token_counter

from dynamiq.prompts import (
    Message,
    MessageRole,
    Prompt,
    VisionMessage,
    VisionMessageTextContent,
    compile_template,
    message_framing_overhead,
)

MODELS = ["gpt-4o", "claude-3-5-sonnet-20241022", "gpt-3.5-turbo"]


def _batched_count(prompt: Prompt, model: str) -> int:
    """What Prompt.count_tokens did before per-message memoization: one batched call."""
    return token_counter(
        model=model, messages=[m.model_dump(exclude={"metadata"}) for m in prompt.messages]
    )


def _conversation(n: int = 20) -> Prompt:
    body = "a typical tool observation with some JSON-ish content. " * 20
    messages = [Message(role=MessageRole.SYSTEM, content="You are an agent. " * 50)]
    for i in range(n):
        messages.append(
            Message(
                role=MessageRole.ASSISTANT,
                content=body,
                tool_calls=[{"id": f"t{i}", "function": {"name": "f", "arguments": "{}"}}],
            )
        )
        messages.append(Message(role=MessageRole.USER, content=body))
    return Prompt(messages=messages)


# --- token counting -------------------------------------------------------------


@pytest.mark.parametrize("model", MODELS)
def test_count_tokens_matches_a_single_batched_call(model):
    """The memoized per-message sum must reproduce the batched count exactly.

    If this drifts, agents start compacting at a different point than before.
    """
    prompt = _conversation()
    assert prompt.count_tokens(model) == _batched_count(prompt, model)


@pytest.mark.parametrize("model", MODELS)
def test_count_tokens_still_matches_after_appending(model):
    prompt = _conversation()
    prompt.count_tokens(model)  # prime the per-message caches

    prompt.messages.append(Message(role=MessageRole.USER, content="a follow-up question"))

    assert prompt.count_tokens(model) == _batched_count(prompt, model)


def test_count_tokens_of_empty_prompt_is_zero():
    """Guards the (len(messages) - 1) overhead correction against an empty list."""
    assert Prompt(messages=[]).count_tokens("gpt-4o") == 0


def test_count_tokens_of_single_message_applies_no_correction():
    prompt = Prompt(messages=[Message(content="hello")])
    assert prompt.count_tokens("gpt-4o") == _batched_count(prompt, "gpt-4o")


def test_count_tokens_handles_vision_messages():
    prompt = Prompt(
        messages=[
            Message(content="describe this"),
            VisionMessage(content=[VisionMessageTextContent(text="some text")]),
        ]
    )
    assert prompt.count_tokens("gpt-4o") == _batched_count(prompt, "gpt-4o")


def test_message_token_cache_invalidates_when_content_changes():
    """A stale count would silently mis-size the context window."""
    message = Message(content="short")
    first = message.count_tokens("gpt-4o")

    message.content = "considerably longer content " * 50
    second = message.count_tokens("gpt-4o")

    assert second > first
    assert second == token_counter(
        model="gpt-4o", messages=[message.model_dump(exclude={"metadata"})]
    )


def test_message_token_cache_invalidates_when_tool_calls_change():
    message = Message(role=MessageRole.ASSISTANT, content="calling a tool")
    first = message.count_tokens("gpt-4o")

    message.tool_calls = [{"id": "t1", "function": {"name": "search", "arguments": '{"q": "x"}'}}]
    second = message.count_tokens("gpt-4o")

    assert second != first


def test_message_token_cache_is_per_model():
    message = Message(content="hello world")
    counts = {model: message.count_tokens(model) for model in MODELS}
    for model, count in counts.items():
        assert count == token_counter(
            model=model, messages=[message.model_dump(exclude={"metadata"})]
        )


@pytest.mark.parametrize("model", MODELS)
def test_framing_overhead_is_measured_not_assumed(model):
    """The correction must come from the tokenizer, not a hardcoded constant."""
    first = {"role": "user", "content": "a"}
    second = {"role": "user", "content": "b"}
    expected = (
        token_counter(model=model, messages=[first])
        + token_counter(model=model, messages=[second])
        - token_counter(model=model, messages=[first, second])
    )
    assert message_framing_overhead(model) == expected


# --- template compilation -------------------------------------------------------


def test_identical_sources_share_one_compiled_template():
    assert compile_template("hello {{ name }}") is compile_template("hello {{ name }}")


def test_different_sources_do_not_share():
    assert compile_template("{{ a }}") is not compile_template("{{ b }}")


def test_rendering_is_unaffected_by_caching():
    message = Message(content="Hello {{ name }}, you are {{ age }}")

    first = message.format_message(name="Ada", age=36)
    second = message.format_message(name="Alan", age=41)

    assert first.content == "Hello Ada, you are 36"
    assert second.content == "Hello Alan, you are 41"


def test_static_messages_are_not_rendered():
    """Static content must pass through untouched, braces and all."""
    prompt = Prompt(messages=[Message(content="literal {{ not_a_var }}", static=True)])
    assert prompt.format_messages()[0]["content"] == "literal {{ not_a_var }}"


def test_message_fields_survive_formatting():
    message = Message(
        role=MessageRole.TOOL,
        content="result for {{ q }}",
        tool_call_id="call-1",
        name="search",
    )
    formatted = message.format_message(q="dynamiq")

    assert formatted.content == "result for dynamiq"
    assert formatted.tool_call_id == "call-1"
    assert formatted.name == "search"
    assert formatted.role is MessageRole.TOOL


def test_compiled_templates_are_safe_to_reuse_across_threads():
    from concurrent.futures import ThreadPoolExecutor

    logging.disable(logging.CRITICAL)
    message = Message(content="value={{ i }}")
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda i: message.format_message(i=i).content, range(200)))

    assert results == [f"value={i}" for i in range(200)]
