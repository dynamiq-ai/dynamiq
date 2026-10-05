import asyncio
from unittest.mock import MagicMock, patch

import pytest

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.runnables import RunnableConfig
from dynamiq.types.cancellation import CanceledException


class Stream:
    def __init__(self, failure=None):
        self.failure = failure
        self.closed = False
        self.completion_stream = self

    def __iter__(self):
        yield MagicMock()
        if self.failure is not None:
            raise self.failure

    async def __aiter__(self):
        yield MagicMock()
        if self.failure is not None:
            raise self.failure

    def close(self):
        self.closed = True

    async def aclose(self):
        self.closed = True


@pytest.fixture
def llm_node():
    with patch("litellm.completion"), patch("litellm.stream_chunk_builder"):
        node = OpenAI(model="gpt-4o-mini", connection=OpenAIConnection(api_key="test-key"))
    node._stream_chunk_builder = MagicMock(return_value=MagicMock())
    node._handle_completion_response = MagicMock(return_value={"content": "ok"})
    return node


@pytest.mark.parametrize("failure", [None, RuntimeError("provider failed"), CanceledException()])
def test_sync_stream_is_closed(llm_node, failure):
    stream = Stream(failure)
    if failure is None:
        assert llm_node._handle_streaming_completion_response(stream, [], RunnableConfig()) == {"content": "ok"}
    else:
        with pytest.raises(type(failure)):
            llm_node._handle_streaming_completion_response(stream, [], RunnableConfig())
    assert stream.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, RuntimeError("provider failed"), asyncio.CancelledError()])
async def test_async_stream_is_closed(llm_node, failure):
    stream = Stream(failure)
    if failure is None:
        assert await llm_node._handle_streaming_completion_response_async(stream, [], RunnableConfig()) == {
            "content": "ok"
        }
    else:
        with pytest.raises(type(failure)):
            await llm_node._handle_streaming_completion_response_async(stream, [], RunnableConfig())
    assert stream.closed


@pytest.mark.parametrize("is_async", [False, True])
def test_cancellation_token_closes_stream(llm_node, is_async):
    config = RunnableConfig()
    config.cancellation.token.cancel()
    stream = Stream()
    with pytest.raises(CanceledException):
        if is_async:
            asyncio.run(llm_node._handle_streaming_completion_response_async(stream, [], config))
        else:
            llm_node._handle_streaming_completion_response(stream, [], config)
    assert stream.closed


def test_close_error_preserves_original_failure(llm_node):
    stream = Stream()
    stream.close = MagicMock(side_effect=RuntimeError("close failed"))
    config = RunnableConfig()
    config.cancellation.token.cancel()
    with pytest.raises(CanceledException):
        llm_node._handle_streaming_completion_response(stream, [], config)
    stream.close.assert_called_once()


@pytest.mark.asyncio
async def test_aclose_error_preserves_original_failure(llm_node):
    stream = Stream()
    called = False

    async def failed_close():
        nonlocal called
        called = True
        raise RuntimeError("close failed")

    stream.aclose = failed_close
    config = RunnableConfig()
    config.cancellation.token.cancel()
    with pytest.raises(CanceledException):
        await llm_node._handle_streaming_completion_response_async(stream, [], config)
    assert called


@pytest.mark.parametrize("is_async", [False, True])
def test_callback_failure_closes_stream(llm_node, is_async):
    llm_node.streaming.enabled = True
    llm_node.run_on_node_execute_stream = MagicMock(side_effect=ValueError("callback failed"))
    stream = Stream()
    with pytest.raises(ValueError, match="callback failed"):
        if is_async:
            asyncio.run(llm_node._handle_streaming_completion_response_async(stream, [], RunnableConfig()))
        else:
            llm_node._handle_streaming_completion_response(stream, [], RunnableConfig())
    assert stream.closed


@pytest.mark.asyncio
async def test_task_cancellation_while_waiting_closes_stream(llm_node):
    started = asyncio.Event()

    class WaitingStream(Stream):
        async def __aiter__(self):
            started.set()
            await asyncio.Event().wait()
            yield MagicMock()

    stream = WaitingStream()
    task = asyncio.create_task(llm_node._handle_streaming_completion_response_async(stream, [], RunnableConfig()))
    await asyncio.wait_for(started.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stream.closed
