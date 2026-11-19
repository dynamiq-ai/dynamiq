import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from litellm import ModelResponse

from dynamiq.callbacks.base import BaseCallbackHandler
from dynamiq.connections import Dynamiq as DynamiqConnection
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.llms import Dynamiq, OpenAI
from dynamiq.nodes.llms.base import LLMContentFilteredError
from dynamiq.prompts import Message, Prompt
from dynamiq.runnables import RunnableConfig
from dynamiq.types.cancellation import CanceledException


class Recorder(BaseCallbackHandler):
    def __init__(self):
        self.calls = []
        self.usage = []
        self.usage_ids = []

    def on_node_execute_run(self, serialized, **kwargs):
        if call := kwargs.get("model_call"):
            self.calls.append(call)
        if usage := kwargs.get("usage_data"):
            self.usage.append(usage)
            self.usage_ids.append(kwargs.get("model_call_id"))


class Stream:
    def __init__(self, failure=None, cancel=None):
        self.failure = failure
        self.cancel = cancel
        self.closed = False

    def __iter__(self):
        yield MagicMock()
        if self.cancel:
            self.cancel.cancel()
            yield MagicMock()
        if self.failure is not None:
            raise self.failure

    async def __aiter__(self):
        for chunk in self:
            yield chunk

    def close(self):
        self.closed = True

    async def aclose(self):
        self.closed = True


def node():
    return OpenAI(
        connection=OpenAIConnection(api_key="test"),
        model="gpt-4o-mini",
        prompt=Prompt(messages=[Message(role="user", content="private-fixture")]),
        error_handling={"max_retries": 0},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("failure", ["provider", "cancel_token", "cancel_task", "builder", "full_response", "usage"])
async def test_response_failure_closes_observed_attempt_without_repeating_transport(mocker, asynchronous, failure):
    llm = node()
    recorder = Recorder()
    config = RunnableConfig(callbacks=[recorder])
    original = asyncio.CancelledError() if failure == "cancel_task" else RuntimeError("private-provider-detail")
    stream = None
    if failure in {"provider", "cancel_token", "cancel_task", "builder"}:
        llm._force_stream = True
        stream = Stream(
            failure=original if failure in {"provider", "cancel_task"} else None,
            cancel=config.cancellation.token if failure == "cancel_token" else None,
        )
        response = stream
        if failure == "builder":
            mocker.patch.object(llm, "_build_streamed_response", side_effect=original)
    elif failure == "full_response":
        response = SimpleNamespace(choices=[])
    else:
        response = ModelResponse(choices=[{"message": {"content": "valid-output"}}])
        mocker.patch.object(llm, "get_usage_data", side_effect=original)
    transport = AsyncMock(return_value=response) if asynchronous else MagicMock(return_value=response)
    mocker.patch.object(llm, "_acall_completion" if asynchronous else "_call_completion", transport)
    expected = (
        CanceledException if failure == "cancel_token" else IndexError if failure == "full_response" else type(original)
    )
    with pytest.raises(expected) as raised:
        if asynchronous:
            await llm.execute_async({}, config=config)
        else:
            llm.execute({}, config=config)
    if failure not in {"cancel_token", "full_response"}:
        assert raised.value is original
    assert transport.call_count == 1
    assert [call["status"] for call in recorder.calls] == ["started", "failed"]
    assert len({call["id"] for call in recorder.calls}) == 1
    assert recorder.usage == []
    assert "private" not in str(recorder.calls)
    if stream is not None:
        assert stream.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("failure", ["content_filter", "router_normalization"])
async def test_failure_after_observed_usage_keeps_known_zero_cost_and_same_attempt(mocker, asynchronous, failure):
    llm = (
        node()
        if failure == "content_filter"
        else Dynamiq(
            connection=DynamiqConnection(api_key="test", router_url="https://router.test/v1"),
            model="unpriced-fixture",
            model_info=Dynamiq.DEFAULT_MODEL_INFO,
            prompt=Prompt(messages=[Message(role="user", content="private-fixture")]),
        )
    )
    response = ModelResponse(
        choices=[{"message": {"content": ""}, "finish_reason": "content_filter"}],
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0},
    )
    if failure == "router_normalization":
        response.choices[0].finish_reason = "stop"
        response.choices[0].message.content = "valid-output"
        response.choices[0].message.reasoning_details = ["invalid-detail"]
    recorder = Recorder()
    transport = AsyncMock(return_value=response) if asynchronous else MagicMock(return_value=response)
    mocker.patch.object(llm, "_acall_completion" if asynchronous else "_call_completion", transport)
    expected = LLMContentFilteredError if failure == "content_filter" else TypeError
    with pytest.raises(expected):
        if asynchronous:
            await llm.execute_async({}, config=RunnableConfig(callbacks=[recorder]))
        else:
            llm.execute({}, config=RunnableConfig(callbacks=[recorder]))
    assert transport.call_count == 1
    assert [call["status"] for call in recorder.calls] == ["started", "failed"]
    assert recorder.usage_ids == [recorder.calls[0]["id"]] == [recorder.calls[1]["id"]]
    assert len(recorder.usage) == 1
    assert recorder.usage[0]["total_tokens_cost_usd"] == 0.0
    assert recorder.usage[0]["complete"] is True


@pytest.mark.asyncio
async def test_cancel_waiting_model_stream_closes_the_same_observed_attempt(mocker):
    llm = node()
    llm._force_stream = True
    recorder = Recorder()
    entered = asyncio.Event()

    class WaitingStream(Stream):
        async def __aiter__(self):
            entered.set()
            await asyncio.Event().wait()
            yield MagicMock()

    stream = WaitingStream()
    transport = mocker.patch.object(llm, "_acall_completion", AsyncMock(return_value=stream))
    task = asyncio.create_task(llm.execute_async({}, config=RunnableConfig(callbacks=[recorder])))
    await asyncio.wait_for(entered.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert transport.call_count == 1
    assert stream.closed
    assert [call["status"] for call in recorder.calls] == ["started", "failed"]
    assert len({call["id"] for call in recorder.calls}) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
async def test_successful_response_keeps_usage_and_business_output(mocker, asynchronous, streaming):
    llm = node()
    recorder = Recorder()
    config = RunnableConfig(callbacks=[recorder])
    completion = ModelResponse(
        choices=[{"message": {"content": "valid-output"}}],
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    )
    stream = Stream() if streaming else None
    response = stream if streaming else completion
    llm._force_stream = streaming
    if streaming:
        mocker.patch.object(llm, "_build_streamed_response", return_value=completion)
    transport = AsyncMock(return_value=response) if asynchronous else MagicMock(return_value=response)
    mocker.patch.object(llm, "_acall_completion" if asynchronous else "_call_completion", transport)
    result = await llm.execute_async({}, config=config) if asynchronous else llm.execute({}, config=config)
    assert result == {"content": "valid-output"}
    assert transport.call_count == 1
    assert [call["status"] for call in recorder.calls] == ["started"]
    assert len(recorder.usage) == 1
    assert recorder.usage[0]["complete"] is True
    assert recorder.usage[0]["total_tokens"] == 0
    if stream is not None:
        assert stream.closed
