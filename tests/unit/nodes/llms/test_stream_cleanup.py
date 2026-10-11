import asyncio
import gc
import json
import socketserver
import subprocess  # nosec B404
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpcore
import litellm
import pytest

from dynamiq.callbacks.streaming import StreamingIteratorCallbackHandler
from dynamiq.connections import Anthropic as AnthropicConnection
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.llms.anthropic import Anthropic
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.prompts import Message, Prompt
from dynamiq.runnables import RunnableConfig, RunnableStatus
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


# A Messages API stream as Anthropic sends it: chunked SSE on a keep-alive connection, with the body's final
# zero-length chunk written a moment after message_stop. LiteLLM stops reading at message_stop, so that chunk is
# still unread when the call returns.
ANTHROPIC_EVENTS = (
    {
        "type": "message_start",
        "message": {
            "id": "msg_local",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 1},
        },
    },
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello from the local server"}},
    {"type": "content_block_stop", "index": 0},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 6},
    },
    {"type": "message_stop"},
)


class AnthropicStreamHandler(socketserver.StreamRequestHandler):
    def handle(self):
        while self.rfile.readline():
            length = 0
            while (header := self.rfile.readline()) not in (b"\r\n", b""):
                name, _, value = header.decode().partition(":")
                if name.lower() == "content-length":
                    length = int(value)
            self.rfile.read(length)
            self.wfile.write(
                b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n"
            )
            for event in ANTHROPIC_EVENTS:
                body = f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()
                self.wfile.write(b"%x\r\n%s\r\n" % (len(body), body))
            time.sleep(0.05)
            self.wfile.write(b"0\r\n\r\n")


class AnthropicServer(socketserver.ThreadingTCPServer):
    daemon_threads = True
    block_on_close = False


@pytest.fixture
def anthropic_api(monkeypatch):
    """A local Messages API that LiteLLM's Anthropic provider is pointed at."""
    server = AnthropicServer(("127.0.0.1", 0), AnthropicStreamHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address
    monkeypatch.setenv("ANTHROPIC_API_BASE", f"http://{host}:{port}")
    yield port
    server.shutdown()
    server.server_close()


def run_anthropic_stream():
    llm = Anthropic(
        model="claude-sonnet-4-5",
        connection=AnthropicConnection(api_key="test-key"),
        streaming={"enabled": True},
        max_tokens=64,
    )
    return llm.run(
        input_data={},
        prompt=Prompt(messages=[Message(role="user", content="Say hello.")]),
        # The node streams only when a streaming callback is listening, as when the runtime serves a run.
        config=RunnableConfig(callbacks=[StreamingIteratorCallbackHandler()]),
    )


def test_anthropic_stream_releases_its_connection(anthropic_api):
    origin = httpcore.Origin(b"http", b"127.0.0.1", anthropic_api)
    pool = litellm.module_level_client.client._transport._pool
    # An unclosed stream sits in a reference cycle; a collection during the test would release it and hide the leak.
    gc.disable()
    try:
        result = run_anthropic_stream()
        in_use = [conn for conn in pool.connections if conn.can_handle_request(origin) and not conn.is_idle()]
    finally:
        gc.enable()

    assert result.status == RunnableStatus.SUCCESS
    assert result.output["content"] == "Hello from the local server"
    assert in_use == []


# Runs in a fresh interpreter, because the failure it guards against deadlocks a thread inside gc.collect(), and the
# interpreter can't collect garbage again after that.
COLLECT_UNDER_POOL_LOCK = """
import gc
import os
import threading
import time

import httpcore._sync.connection_pool as connection_pool

from tests.unit.nodes.llms.test_stream_cleanup import run_anthropic_stream

gc.disable()  # keeps the first call's stream, if it leaked, uncollected until the collection below
run_anthropic_stream()
time.sleep(0.5)  # LiteLLM's success logging holds the stream in a background thread for a moment

create_request = connection_pool.PoolRequest.__init__


def collect_under_pool_lock(self, request):
    # ConnectionPool.handle_request creates its PoolRequest while holding the pool lock. In production, an allocation
    # there started a collection that finalized a leaked stream, and that stream's close takes the same lock.
    connection_pool.PoolRequest.__init__ = create_request
    gc.collect()
    create_request(self, request)


connection_pool.PoolRequest.__init__ = collect_under_pool_lock
call = threading.Thread(target=run_anthropic_stream, daemon=True)
call.start()
call.join(timeout=20)
print("deadlocked" if call.is_alive() else "finished", flush=True)
os._exit(0)
"""


def test_collection_under_the_pool_lock_does_not_deadlock(anthropic_api):
    result = subprocess.run(  # nosec B603
        [sys.executable, "-c", COLLECT_UNDER_POOL_LOCK],
        cwd=Path(__file__).parents[4],  # the repository root, so the script can import this module
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.stdout.split()[-1:] == ["finished"], result.stderr[-3000:]
