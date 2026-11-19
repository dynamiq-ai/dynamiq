"""The shared stream assembler must not label its own token estimates as reported usage."""

import copy
from unittest.mock import patch

import pytest
from litellm.types.utils import ModelResponseStream

from dynamiq.connections import AWS
from dynamiq.nodes.llms import Bedrock
from dynamiq.prompts import Prompt
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types.streaming import StreamingConfig
from tests.unit.nodes.llms.test_openai_native_sdk import Collector, FakeOpenAI, make_node


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("reported", [False, True])
async def test_litellm_public_stream_keeps_missing_usage_distinct_from_reported_usage(asynchronous, reported):
    fake = FakeOpenAI()
    node = make_node(
        fake,
        native=False,
        streaming=StreamingConfig(enabled=True, include_usage=reported),
        num_retries=0,
    )
    collector = Collector()
    # The ordinary LiteLLM async path owns its client; inject only the mock transport client.
    acompletion = node._acompletion

    async def send(**params):
        return await acompletion(**params, client=fake.async_client())

    node._acompletion = send
    with patch("socket.socket.connect", side_effect=AssertionError("external network forbidden")):
        result = (
            await node.run_async({}, config=RunnableConfig(callbacks=[collector]))
            if asynchronous
            else node.run_sync({}, config=RunnableConfig(callbacks=[collector]))
        )

    assert result.status == RunnableStatus.SUCCESS, result.error
    assert result.output == {"content": "Hello"}
    assert len(fake.requests) == 1
    assert fake.requests[0]["stream"] is True
    assert collector.usage["complete"] is reported
    if reported:
        assert (collector.usage["prompt_tokens"], collector.usage["completion_tokens"]) == (11, 7)
    else:
        # Compatibility estimates and their cost remain available, but cannot settle observed usage.
        assert collector.usage["prompt_tokens"] > 0
        assert collector.usage["completion_tokens"] > 0
    assert collector.usage["total_tokens_cost_usd"] > 0
    assert len(collector.model_calls) == 1
    assert collector.model_calls[0]["status"] == "started"
    assert "_dynamiq_usage_observed" not in str(result.to_dict())


class NormalizedStream:
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    def __iter__(self):
        return iter(self.chunks)

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk

    def close(self):
        self.closed = True

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "usages, complete, expected",
    [
        ([], False, None),
        ([{"prompt_tokens": 11, "completion_tokens": 7}], True, (11, 7)),
        ([{"prompt_tokens": 0, "completion_tokens": 0}], False, None),
        ([{"prompt_tokens": 11, "completion_tokens": 0}], False, None),
        ([{"prompt_tokens": True, "completion_tokens": 7}], False, None),
        ([{"prompt_tokens": 11.0, "completion_tokens": 7}], False, None),
        ([{"prompt_tokens": -1, "completion_tokens": 7}], False, None),
        ([{"prompt_tokens": 11, "completion_tokens": 3}, {"prompt_tokens": 11, "completion_tokens": 7}], True, (11, 7)),
        (
            [{}, {"prompt_tokens": 11, "completion_tokens": 0}, {"prompt_tokens": 0, "completion_tokens": 7}],
            True,
            (11, 7),
        ),
    ],
    ids=[
        "missing",
        "reported",
        "zero-replaced-by-builder",
        "partial-replaced-by-builder",
        "boolean",
        "float",
        "negative",
        "cumulative",
        "split-counters",
    ],
)
async def test_normalized_bedrock_stream_does_not_present_builder_estimates_as_observed(
    asynchronous, usages, complete, expected
):
    node = Bedrock(
        model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
        connection=AWS(access_key_id="k", secret_access_key="s", region="us-east-1"),
        prompt=Prompt(messages=[{"role": "user", "content": "hi"}]),
        streaming=StreamingConfig(enabled=True),
        num_retries=0,
    )
    base = {"id": "normalized-bedrock", "created": 1, "model": node.model}
    chunks = [
        ModelResponseStream(**base, choices=[{"index": 0, "delta": {"content": "Hello"}}]),
        ModelResponseStream(**base, choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}]),
    ]
    for usage in usages:
        chunk = ModelResponseStream(**base, choices=[], usage=copy.deepcopy(usage))
        # Preserve the SDK-visible malformed types in these fixtures, before the stream builder normalizes them.
        for field, value in usage.items():
            object.__setattr__(chunk.usage, field, value)
        chunks.append(chunk)
    stream = NormalizedStream(chunks)
    collector = Collector()
    node._completion = lambda **_: stream

    async def send(**_):
        return stream

    node._acompletion = send
    with patch("socket.socket.connect", side_effect=AssertionError("external network forbidden")):
        result = (
            await node.run_async({}, config=RunnableConfig(callbacks=[collector]))
            if asynchronous
            else node.run_sync({}, config=RunnableConfig(callbacks=[collector]))
        )

    assert result.status == RunnableStatus.SUCCESS, result.error
    assert result.output == {"content": "Hello"}
    assert stream.closed
    assert collector.usage["complete"] is complete
    if expected:
        assert (collector.usage["prompt_tokens"], collector.usage["completion_tokens"]) == expected
    else:
        assert collector.usage["prompt_tokens"] > 0
        assert collector.usage["completion_tokens"] > 0
    # The fixture reports no provider USD price; the new confidence flag does not invent one.
    assert collector.usage["total_tokens_cost_usd"] is None
