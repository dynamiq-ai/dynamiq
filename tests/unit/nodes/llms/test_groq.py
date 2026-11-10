import asyncio
import functools
import json
from unittest.mock import MagicMock

import httpx
import litellm
import pytest
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

from dynamiq.callbacks.streaming import StreamingIteratorCallbackHandler
from dynamiq.connections import Groq as GroqConnection
from dynamiq.nodes.llms.groq import Groq
from dynamiq.prompts import Prompt
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus
from dynamiq.types.streaming import StreamingConfig


def _groq_error(code: str, failed_generation: str) -> Exception:
    body = {"error": {"message": "Please adjust your prompt.", "code": code, "failed_generation": failed_generation}}
    return Exception(f"litellm.BadRequestError: GroqException - {json.dumps(body)}")


def _run(node: Groq, completion, callbacks=()) -> tuple[dict, list[dict]]:
    sent = []

    def record(**params):
        sent.append(params)
        return completion(**params)

    node._completion = record
    result = node.execute(MagicMock(messages=None, files=None), config=RunnableConfig(callbacks=list(callbacks)))
    return result, sent


def _node(**kwargs) -> Groq:
    return Groq(
        model="groq/openai/gpt-oss-120b",
        connection=GroqConnection(api_key="x"),
        prompt=Prompt(messages=[{"role": "user", "content": "Fetch invoice INV-88."}]),
        **kwargs,
    )


@pytest.mark.parametrize("code", ["json_validate_failed", "tool_use_failed"])
def test_a_reply_groq_rejects_reaches_the_caller_as_text(code):
    reply = '{"vendor": "Nordwind GmbH", "total_eur": 1234.5}'

    def completion(**params):
        raise _groq_error(code, reply)

    result, _ = _run(_node(), completion)

    assert result["content"] == reply


def test_a_streamed_request_groq_rejects_as_it_is_sent_is_sent_again_without_streaming():
    """LiteLLM sends a streamed request that has a response format without streaming, so Groq rejects it up front."""
    reply = "Your locker code is 6360CF."

    def completion(**params):
        raise _groq_error("tool_use_failed", reply)

    result, sent = _run(
        _node(streaming=StreamingConfig(enabled=True)), completion, callbacks=[StreamingIteratorCallbackHandler()]
    )

    assert result["content"] == reply
    assert [params.get("stream") for params in sent] == [True, False]


def test_another_bad_request_still_fails():
    def completion(**params):
        raise Exception('litellm.BadRequestError: GroqException - {"error": {"code": "model_not_found"}}')

    with pytest.raises(Exception, match="model_not_found"):
        _run(_node(), completion)


def _text(content: str) -> dict:
    delta = {"role": "assistant", "content": content}
    return {"id": "chatcmpl-1", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": delta}]}


def _run_stream(mode: str, events: list[dict]) -> RunnableResult:
    """Run the node over ``events`` served as Groq's event stream, which LiteLLM's own Groq code reads."""
    body = "".join(
        ("event: error\n" if "error" in event else "") + f"data: {json.dumps(event)}\n\n" for event in events
    )
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)
    )
    config = RunnableConfig(callbacks=[StreamingIteratorCallbackHandler()])
    streaming = StreamingConfig(enabled=True)
    if mode == "sync":
        client = HTTPHandler(client=httpx.Client(transport=transport))
        return _node(streaming=streaming, client=client).run(input_data={}, config=config)
    client = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=transport)
    node = _node(streaming=streaming)
    node._acompletion = functools.partial(litellm.acompletion, client=client)
    return asyncio.run(node.run_async(input_data={}, config=config))


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_a_reply_groq_rejects_while_streaming_it_reaches_the_caller_as_text(mode):
    """Groq streams the reply's text, then rejects it in an error event that carries the tool call it held back."""
    rejection = {
        "message": "attempted to call tool 'get_weather' which was not in request.tools",
        "type": "invalid_request_error",
        "code": "tool_use_failed",
        "failed_generation": '{"name": "get_weather", "arguments": {"city": "Paris"}}',
        "status_code": 400,
    }

    result = _run_stream(mode, [_text("Checking the weather. "), {"error": rejection}])

    assert result.status == RunnableStatus.SUCCESS, result.error
    assert result.output["content"] == 'Checking the weather. {"name": "get_weather", "arguments": {"city": "Paris"}}'


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_another_error_in_a_stream_still_fails(mode):
    error = {"message": "Rate limit reached", "type": "tokens", "code": "rate_limit_exceeded"}

    result = _run_stream(mode, [_text("Checking"), {"error": error}])

    assert result.status == RunnableStatus.FAILURE
    assert "Rate limit reached" in result.error.message
