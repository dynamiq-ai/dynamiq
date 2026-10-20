"""Real SDK parsing, agent history and subsequent HTTP serialization, without network."""

import json
from copy import deepcopy

import httpx
import litellm
import pytest

from dynamiq.callbacks.streaming import StreamingIteratorCallbackHandler
from dynamiq.connections import HttpApiKey
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.agents.base import InferenceMode
from dynamiq.nodes.llms.custom_llm import CustomLLM
from dynamiq.prompts import Message, Prompt
from dynamiq.prompts.prompts import VisionMessage
from dynamiq.runnables import RunnableConfig, RunnableStatus

REASONING = "Check the image before calling the tool."
SIGNATURE = "opaque-signature-123"
CALL = {"id": "call-history", "type": "function", "function": {"name": "lookup", "arguments": '{"q":"test"}'}}
TOOLS = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}}}]


def make_node(protocol, model="MiniMax-M3.1-Flash", url=None, **kwargs):
    return CustomLLM(
        model=model,
        provider_prefix=protocol,
        connection=HttpApiKey(
            api_key="offline-placeholder",
            url=url or "https://api.minimax.io/" + ("anthropic" if protocol == "anthropic" else "v1"),
        ),
        max_tokens=4096,
        prompt=Prompt(messages=[Message(role="user", content="Hello")]),
        **kwargs,
    )


def response_body(body, tools):
    if "anthropic" in body.pop("_protocol", ""):
        content = [{"type": "text", "text": "Done"}]
        if tools:
            content = [
                {"type": "thinking", "thinking": REASONING, "signature": SIGNATURE},
                {"type": "text", "text": "Checking"},
                {"type": "tool_use", "id": CALL["id"], "name": "lookup", "input": {"q": "test"}},
            ]
        return {
            "id": "msg-history",
            "type": "message",
            "role": "assistant",
            "model": body["model"],
            "content": content,
            "stop_reason": "tool_use" if tools else "end_turn",
            "usage": {"input_tokens": 3, "output_tokens": 4},
        }
    message = {"role": "assistant", "content": "Checking" if tools else "Done"}
    if tools:
        message.update(reasoning_content=REASONING, tool_calls=[deepcopy(CALL)])
    return {
        "id": "chat-history",
        "object": "chat.completion",
        "created": 1,
        "model": body["model"],
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if tools else "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
    }


def stream_body(response, protocol):
    if protocol == "anthropic":
        events = [{"type": "message_start", "message": {**response, "content": [], "stop_reason": None}}]
        for index, block in enumerate(response["content"]):
            if block["type"] == "thinking":
                start = {"type": "thinking", "thinking": "", "signature": ""}
                deltas = [
                    {"type": "thinking_delta", "thinking": REASONING},
                    {"type": "signature_delta", "signature": SIGNATURE},
                ]
            elif block["type"] == "tool_use":
                start = {**block, "input": {}}
                deltas = [{"type": "input_json_delta", "partial_json": json.dumps(block["input"])}]
            else:
                start = {"type": "text", "text": ""}
                deltas = [{"type": "text_delta", "text": block["text"]}]
            events.append({"type": "content_block_start", "index": index, "content_block": start})
            events.extend({"type": "content_block_delta", "index": index, "delta": delta} for delta in deltas)
            events.append({"type": "content_block_stop", "index": index})
        events.extend(
            [
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": response["stop_reason"]},
                    "usage": {"output_tokens": 4},
                },
                {"type": "message_stop"},
            ]
        )
        return "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events).encode()
    message = response["choices"][0]["message"]
    deltas = [{"role": "assistant"}]
    if "reasoning_content" in message:
        deltas.append({"reasoning_content": message["reasoning_content"]})
    deltas.append({"content": message["content"]})
    if "tool_calls" in message:
        deltas.append({"tool_calls": [{"index": 0, **CALL}]})
    chunks = [
        {
            "id": response["id"],
            "object": "chat.completion.chunk",
            "created": 1,
            "model": response["model"],
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
        for delta in deltas
    ]
    chunks.append(
        {
            **chunks[0],
            "choices": [{"index": 0, "delta": {}, "finish_reason": response["choices"][0]["finish_reason"]}],
            "usage": response["usage"],
        }
    )
    return ("".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n").encode()


@pytest.fixture
def wire(monkeypatch):
    captured = []
    state = {"tools": False}

    def send(client, request, **kwargs):
        body = json.loads(request.content)
        captured.append((str(request.url), body))
        protocol = "anthropic" if request.url.path.endswith("/messages") else "openai"
        response = response_body({**body, "_protocol": protocol}, state["tools"])
        if body.get("stream"):
            return httpx.Response(
                200,
                content=stream_body(response, protocol),
                headers={"content-type": "text/event-stream"},
                request=request,
            )
        return httpx.Response(200, json=response, request=request)

    async def asend(client, request, **kwargs):
        return send(client, request, **kwargs)

    monkeypatch.setattr(httpx.Client, "send", send)
    monkeypatch.setattr(httpx.AsyncClient, "send", asend)
    return captured, state


@pytest.mark.parametrize("protocol", ["openai", "anthropic", "minimax"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("model", ["MiniMax-M3.1-Flash", "MiniMax-M3"])
def test_reasoning_survives_agent_tool_continuation(wire, protocol, stream, model):
    captured, state = wire
    controls = {"thinking": {"type": "adaptive"}} if protocol == "anthropic" else {}
    node = make_node(protocol, model=model, streaming={"enabled": stream}, **controls)
    node.prompt = Prompt(
        messages=[
            VisionMessage(
                content=[
                    {"type": "text", "text": "Inspect"},
                    {"type": "image_url", "image_url": {"url": "https://example.test/image.png"}},
                ]
            )
        ],
        tools=TOOLS,
    )
    state["tools"] = True
    config = RunnableConfig(callbacks=[StreamingIteratorCallbackHandler()] if stream else [])
    result = node.run(input_data={}, config=config)
    assert result.status == RunnableStatus.SUCCESS, result.error
    assert bool(captured[-1][1].get("stream")) == stream
    assert result.output["reasoning_content"] == REASONING
    assert result.output["tool_calls"][0]["id"] == CALL["id"]
    if protocol == "anthropic":
        assert result.output["thinking_blocks"] == [{"type": "thinking", "thinking": REASONING, "signature": SIGNATURE}]

    agent = Agent(llm=node, inference_mode=InferenceMode.FUNCTION_CALLING)
    agent._prompt = node.prompt.model_copy(deep=True)
    agent._append_assistant_message(result, result.output["content"])
    assistant = agent._prompt.messages[-1]
    assert assistant.reasoning_content == REASONING
    # Exercise checkpoint-style message serialization, not just direct result reuse.
    restored_message = Message.from_dict(assistant.to_dict())
    assert restored_message.format_message().reasoning_content == REASONING
    assert restored_message.format_message().thinking_blocks == assistant.thinking_blocks
    agent._prompt.messages[-1] = restored_message
    agent._prompt.messages.append(Message(role="tool", content="tool result", tool_call_id=CALL["id"]))
    node.prompt = agent._prompt
    state["tools"] = False
    assert node.run(input_data={}, config=config).status == RunnableStatus.SUCCESS
    body = captured[-1][1]
    history = next(message for message in body["messages"] if message["role"] == "assistant")
    assert "image.png" in json.dumps(body["messages"][0])
    assert "tool result" in json.dumps(body["messages"])
    if protocol == "anthropic":
        assert history["content"][0] == {"type": "thinking", "thinking": REASONING, "signature": SIGNATURE}
        assert any(block.get("id") == CALL["id"] for block in history["content"])
    else:
        assert history["reasoning_content"] == REASONING
        assert history["tool_calls"][0]["id"] == CALL["id"]


@pytest.mark.parametrize("optout", [False, True])
def test_official_request_does_not_change_same_custom_node_or_alias(wire, monkeypatch, optout):
    captured, _ = wire
    if optout:
        monkeypatch.setenv("DYNAMIQ_SYNC_MODEL_REGISTRY_TO_LITELLM", "false")
    key = "anthropic/MiniMax-M3.1-Flash"
    monkeypatch.setitem(
        litellm.model_cost,
        key,
        {
            "litellm_provider": "anthropic",
            "max_input_tokens": 1234,
            "input_cost_per_token": 0.123,
            "supports_reasoning": True,
            "supports_adaptive_thinking": False,
        },
    )
    original = deepcopy(litellm.model_cost[key])
    custom = make_node("anthropic", url="https://custom.example/anthropic", thinking={"type": "adaptive"})
    assert custom.run(input_data={}).status == RunnableStatus.SUCCESS
    before = deepcopy(captured[-1])
    official = make_node("anthropic", thinking={"type": "adaptive"}, output_config={"effort": "xhigh"})
    assert official.run(input_data={}).status == RunnableStatus.SUCCESS
    assert captured[-1][1]["thinking"] == {"type": "adaptive"}
    assert captured[-1][1]["output_config"] == {"effort": "xhigh"}
    assert litellm.model_cost[key] == original
    assert custom.run(input_data={}).status == RunnableStatus.SUCCESS
    assert captured[-1] == before
    restored = CustomLLM.model_validate(custom.model_dump())
    assert restored.run(input_data={}).status == RunnableStatus.SUCCESS
    assert captured[-1][0] == before[0]
    for key in ("model", "thinking", "messages"):
        assert captured[-1][1][key] == before[1][key]


@pytest.mark.parametrize("protocol", ["openai", "anthropic", "minimax"])
@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_m3_ignores_effort_before_sdk_validation(wire, protocol, effort):
    captured, _ = wire
    options = {"thinking": {"type": "disabled"}, "output_config": {"effort": effort}}
    if protocol != "anthropic":
        options = {"extra_body": {"thinking": {"type": "disabled"}, "reasoning_effort": effort}}
    node = make_node(protocol, model="MiniMax-M3", **options)
    saved = node.model_dump()
    result = node.run(input_data={})
    assert result.status == RunnableStatus.SUCCESS, result.error
    body = captured[-1][1]
    assert body["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in body
    assert "effort" not in body.get("output_config", {})
    assert node.model_dump() == saved


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_async_anthropic_controls_and_reasoning(wire, stream):
    captured, state = wire
    state["tools"] = True
    node = make_node(
        "anthropic", thinking={"type": "adaptive"}, output_config={"effort": "xhigh"}, streaming={"enabled": stream}
    )
    config = RunnableConfig(callbacks=[StreamingIteratorCallbackHandler()] if stream else [])
    result = await node.run_async(input_data={}, config=config)
    assert result.status == RunnableStatus.SUCCESS, result.error
    assert bool(captured[-1][1].get("stream")) == stream
    assert result.output["thinking_blocks"][0] == {
        "type": "thinking",
        "thinking": REASONING,
        "signature": SIGNATURE,
    }
    assert result.output["reasoning_content"] == REASONING
    assert captured[-1][1]["output_config"] == {"effort": "xhigh"}


@pytest.mark.parametrize("protocol", ["openai", "anthropic", "minimax"])
def test_m3_generic_effort_does_not_enable_thinking(wire, protocol):
    captured, _ = wire
    node = make_node(protocol, model="MiniMax-M3", reasoning_effort="xhigh")
    assert node.run(input_data={}).status == RunnableStatus.SUCCESS
    body = captured[-1][1]
    assert "thinking" not in body
    assert "reasoning_effort" not in body
    assert "output_config" not in body


@pytest.mark.parametrize("model", ["MiniMax-M3.1-Flash", "MiniMax-M3", "MiniMax-M2.7", "custom-deployment"])
def test_custom_route_parameters_are_not_rewritten(wire, model):
    captured, _ = wire
    node = make_node(
        "openai",
        model=model,
        url="https://custom.example/v1",
        extra_body={
            "thinking": {"type": "disabled"},
            "reasoning_effort": "xhigh",
        },
    )
    assert node.run(input_data={}).status == RunnableStatus.SUCCESS
    assert captured[-1][1]["reasoning_effort"] == "xhigh"
    assert captured[-1][1]["thinking"] == {"type": "disabled"}
    assert captured[-1][1]["model"] == model
