"""Exercise the real LLM and SDK serialization with an offline HTTP boundary."""

import json

import httpx
import pytest

from dynamiq.connections import HttpApiKey
from dynamiq.nodes.llms.custom_llm import CustomLLM
from dynamiq.prompts import Message, Prompt
from dynamiq.prompts.prompts import VisionMessage
from dynamiq.runnables import RunnableStatus


@pytest.fixture
def captured_requests(monkeypatch):
    captured = []

    def send(client, request, **kwargs):
        body = json.loads(request.content)
        captured.append((request, body))
        if request.url.path.endswith("/messages"):
            response = {
                "id": "msg-offline",
                "type": "message",
                "role": "assistant",
                "model": body["model"],
                "content": [{"type": "text", "text": "Hello"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 3, "output_tokens": 1},
            }
        else:
            response = {
                "id": "chat-offline",
                "object": "chat.completion",
                "created": 1,
                "model": body["model"],
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            }
        return httpx.Response(200, json=response, request=request)

    monkeypatch.setattr(httpx.Client, "send", send)
    return captured


def run_model(model, region, protocol, prompt=None, **options):
    base = f"https://api.minimax.{region}/" + ("anthropic" if protocol == "anthropic" else "v1")
    node = CustomLLM(
        model=model,
        provider_prefix=protocol,
        connection=HttpApiKey(api_key="offline-placeholder", url=base),
        prompt=prompt or Prompt(messages=[Message(role="user", content="Hello")]),
        max_tokens=4096,
        **options,
    )
    result = node.run(input_data={})
    assert result.status == RunnableStatus.SUCCESS, result.error
    assert result.output["content"] == "Hello"
    assert node.model == model
    return node


@pytest.mark.parametrize("region", ["io", "cn"])
@pytest.mark.parametrize("protocol", ["openai", "anthropic", "minimax"])
@pytest.mark.parametrize("model", ["MiniMax-M3.1-Flash", "MiniMax-M3"])
def test_minimax_defaults_reach_http(captured_requests, region, protocol, model):
    run_model(model, region, protocol)
    assert len(captured_requests) == 1
    request, body = captured_requests[0]
    assert request.url.host == f"api.minimax.{region}"
    expected_path = "/anthropic/v1/messages" if protocol == "anthropic" else "/v1/chat/completions"
    assert request.url.path == expected_path
    assert body["model"] == model
    assert body["messages"][0]["role"] == "user"
    assert "service_tier" not in body
    assert "reasoning_split" not in body
    # Omitted controls retain each protocol's documented server defaults.
    assert "thinking" not in body
    assert "reasoning_effort" not in body
    assert "output_config" not in body


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_minimax_flash_explicit_effort(captured_requests, protocol, effort):
    extra = {"thinking": {"type": "adaptive"}}
    if protocol == "anthropic":
        extra["output_config"] = {"effort": effort}
    else:
        extra["reasoning_effort"] = effort
    options = extra if protocol == "anthropic" else {"extra_body": extra}
    run_model("MiniMax-M3.1-Flash", "io", protocol, **options)
    body = captured_requests[0][1]
    for key, value in extra.items():
        assert body[key] == value


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
@pytest.mark.parametrize("model", ["MiniMax-M3.1-Flash", "MiniMax-M3", "MiniMax-M2.7", "custom-deployment"])
def test_saved_model_ids_are_preserved(captured_requests, model, protocol):
    node = run_model(model, "cn", protocol)
    assert captured_requests[0][1]["model"] == model
    restored = CustomLLM.model_validate(node.model_dump())
    assert restored.model == model
    assert restored.run(input_data={}).status == RunnableStatus.SUCCESS
    assert len(captured_requests) == 2
    assert captured_requests[-1][1]["model"] == model


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
@pytest.mark.parametrize("thinking", ["adaptive", "disabled"])
def test_minimax_m3_thinking_is_preserved(captured_requests, protocol, thinking):
    controls = {"thinking": {"type": thinking}}
    options = controls if protocol == "anthropic" else {"extra_body": controls}
    run_model("MiniMax-M3", "io", protocol, **options)
    assert captured_requests[0][1]["thinking"] == {"type": thinking}


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
def test_minimax_image_content_survives_serialization(captured_requests, protocol):
    image_url = "https://example.test/image.png"
    prompt = Prompt(
        messages=[
            VisionMessage(
                content=[
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]
            )
        ]
    )
    run_model("MiniMax-M3.1-Flash", "io", protocol, prompt=prompt)
    content = captured_requests[0][1]["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "Describe this image"}
    image = content[1]
    if protocol == "anthropic":
        assert image["type"] == "image"
        assert image["source"] == {"type": "url", "url": image_url}
    else:
        assert image["type"] == "image_url"
        assert image["image_url"]["url"] == image_url


@pytest.mark.parametrize(
    ("model", "url"),
    [
        ("MiniMax-M3.1-Flash", "https://example.test/anthropic"),
        ("MiniMax-M3.1-Flash", "https://api.minimax.io/v1"),
        ("MiniMax-M2.7", "https://api.minimax.io/anthropic"),
        ("custom-deployment", "https://api.minimax.io/anthropic"),
    ],
)
def test_compatibility_metadata_does_not_register_other_routes(mocker, model, url):
    import litellm

    register = mocker.spy(litellm, "register_model")
    node = CustomLLM(
        model=model,
        provider_prefix="anthropic",
        connection=HttpApiKey(api_key="offline-placeholder", url=url),
    )
    assert node.update_completion_params({"model": model})["model"] == f"anthropic/{model}"
    register.assert_not_called()


def test_compatibility_metadata_honors_sync_opt_out(mocker, monkeypatch):
    import litellm

    monkeypatch.setenv("DYNAMIQ_SYNC_MODEL_REGISTRY_TO_LITELLM", "false")
    register = mocker.spy(litellm, "register_model")
    node = CustomLLM(
        model="MiniMax-M3.1-Flash",
        provider_prefix="anthropic",
        connection=HttpApiKey(api_key="offline-placeholder", url="https://api.minimax.io/anthropic"),
    )
    node.update_completion_params({"model": node.model})
    register.assert_not_called()
