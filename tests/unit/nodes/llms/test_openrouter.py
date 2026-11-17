import asyncio
import functools
import json

import httpx
import litellm
import pytest
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

from dynamiq.callbacks.streaming import StreamingIteratorCallbackHandler
from dynamiq.connections import OpenRouter as OpenRouterConnection
from dynamiq.nodes.llms import OpenRouter
from dynamiq.prompts import Message, Prompt
from dynamiq.runnables import RunnableConfig
from dynamiq.types.streaming import StreamingConfig

REPLY = {"id": "gen-1", "created": 1794324307, "model": "openai/gpt-5.4-mini"}
TOOL_CALL = {
    "index": 0,
    "id": "call_1",
    "type": "function",
    "function": {"name": "write_file", "arguments": '{"path": "poem.txt", "content": "The sea'},
}


def _run(mode: str, streamed: bool, native_finish_reason: str) -> dict:
    """The node's output for a tool call as OpenRouter returns it, read by LiteLLM's own OpenRouter code."""
    finish = {"finish_reason": "tool_calls", "native_finish_reason": native_finish_reason}
    message = {"role": "assistant", "content": None, "tool_calls": [TOOL_CALL]}

    def openrouter(request: httpx.Request) -> httpx.Response:
        if not streamed:
            return httpx.Response(200, json={**REPLY, "choices": [{"index": 0, "message": message, **finish}]})
        chunks = [{"index": 0, "delta": message, "finish_reason": None}, {"index": 0, "delta": {}, **finish}]
        events = (f"data: {json.dumps({**REPLY, 'choices': [chunk]})}\n\n" for chunk in chunks)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content="".join(events))

    transport = httpx.MockTransport(openrouter)
    node = OpenRouter(
        model="openai/gpt-5.4-mini",
        connection=OpenRouterConnection(api_key="x"),
        prompt=Prompt(messages=[Message(role="user", content="Write a poem to poem.txt.")]),
        streaming=StreamingConfig(enabled=streamed),
        client=HTTPHandler(client=httpx.Client(transport=transport)),
    )
    config = RunnableConfig(callbacks=[StreamingIteratorCallbackHandler()] if streamed else [])
    if mode == "sync":
        return node.run(input_data={}, config=config).output
    client = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=transport)
    node._acompletion = functools.partial(litellm.acompletion, client=client)
    return asyncio.run(node.run_async(input_data={}, config=config)).output


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("streamed", [False, True], ids=["whole", "streamed"])
@pytest.mark.parametrize("native_finish_reason", ["length", "max_output_tokens"])
def test_a_tool_call_cut_off_at_max_tokens_is_reported_as_cut_off(mode, streamed, native_finish_reason):
    """OpenRouter ends such a reply with "tool_calls", and gives the provider's reason as native_finish_reason."""
    assert _run(mode, streamed, native_finish_reason)["finish_reason"] == "length"


@pytest.mark.parametrize("streamed", [False, True], ids=["whole", "streamed"])
def test_a_complete_tool_call_is_not_reported_as_cut_off(streamed):
    output = _run("sync", streamed, "tool_calls")

    assert "finish_reason" not in output
    assert [call["function"]["name"] for call in output["tool_calls"]] == ["write_file"]
