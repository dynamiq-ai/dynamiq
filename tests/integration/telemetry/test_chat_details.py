"""Provider names, response format, cache usage and inline images on chat spans."""

import base64

import pytest
from litellm import ModelResponse
from litellm.types.utils import PromptTokensDetailsWrapper, Usage

from dynamiq import Workflow, connections
from dynamiq.flows import Flow
from dynamiq.nodes import llms
from dynamiq.prompts import (
    Message,
    MessageRole,
    Prompt,
    VisionMessage,
    VisionMessageImageContent,
    VisionMessageImageURL,
    VisionMessageTextContent,
)
from dynamiq.runnables import RunnableStatus
from tests.integration.telemetry.conftest import attrs_json

PROVIDERS = [
    (llms.OpenAI, connections.OpenAI, "gpt-4o", "openai", "gpt-4o"),
    (llms.Anthropic, connections.Anthropic, "claude-sonnet-4-5", "anthropic", "claude-sonnet-4-5"),
    (llms.Gemini, connections.Gemini, "gemini-2.0-flash", "gcp.gemini", "gemini-2.0-flash"),
    (llms.Mistral, connections.Mistral, "mistral-large-latest", "mistral_ai", "mistral-large-latest"),
    (llms.Groq, connections.Groq, "llama-3.3-70b-versatile", "groq", "llama-3.3-70b-versatile"),
    (llms.xAI, connections.xAI, "grok-3", "x_ai", "grok-3"),
    (llms.TogetherAI, connections.TogetherAI, "meta-llama/Llama-3-70b", "together_ai", "meta-llama/Llama-3-70b"),
    (llms.OpenRouter, connections.OpenRouter, "openai/gpt-4o", "openrouter", "openai/gpt-4o"),
]


def llm_of(cls, connection_cls, model, **kwargs):
    return cls(
        connection=connection_cls(api_key="test-key"),
        model=model,
        prompt=Prompt(messages=[Message(role=MessageRole.USER, content="Say hi")]),
        **kwargs,
    )


@pytest.mark.parametrize(("cls", "connection_cls", "model", "expected_provider", "request_model"), PROVIDERS)
def test_provider_and_model_names(
    mock_llm_executor, config, trace, cls, connection_cls, model, expected_provider, request_model
):
    node = llm_of(cls, connection_cls, model)

    result = Workflow(flow=Flow(nodes=[node])).run(input_data={}, config=config)

    assert result.status == RunnableStatus.SUCCESS
    spans = trace()
    (chat,) = (s for s in spans.spans if s.attributes.get("gen_ai.operation.name") == "chat")
    assert chat.name == f"chat {request_model}"
    assert chat.attributes["gen_ai.provider.name"] == expected_provider
    assert chat.attributes["gen_ai.system"] == expected_provider
    assert chat.attributes["gen_ai.request.model"] == request_model
    # the model as configured keeps the provider prefix Dynamiq adds
    assert chat.attributes["dynamiq.node.model"] == node.model


def test_json_response_format_is_reported(mock_llm_executor, config, trace):
    node = llm_of(llms.OpenAI, connections.OpenAI, "gpt-4o", response_format={"type": "json_object"})

    Workflow(flow=Flow(nodes=[node])).run(input_data={}, config=config)

    assert trace().one("chat gpt-4o").attributes["gen_ai.output.type"] == "json"


def test_cache_token_usage_is_reported_under_both_names(mocker, config, trace):
    reply = ModelResponse(
        choices=[{"message": {"content": "hi"}, "finish_reason": "length"}],
        usage=Usage(
            prompt_tokens=20,
            completion_tokens=5,
            total_tokens=25,
            prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=8),
            cache_creation_input_tokens=3,
        ),
    )
    mocker.patch("dynamiq.nodes.llms.base.BaseLLM._completion", return_value=reply)
    node = llm_of(llms.Anthropic, connections.Anthropic, "claude-sonnet-4-5")

    Workflow(flow=Flow(nodes=[node])).run(input_data={}, config=config)

    a = trace().one("chat claude-sonnet-4-5").attributes
    assert a["gen_ai.usage.input_tokens"] == 20  # cached tokens included
    assert a["gen_ai.usage.output_tokens"] == 5
    assert a["gen_ai.usage.cache_read.input_tokens"] == 8
    assert a["gen_ai.usage.cache_creation.input_tokens"] == 3
    assert a["gen_ai.usage.cache_write.input_tokens"] == 3
    assert list(a["gen_ai.response.finish_reasons"]) == ["length"]


@pytest.fixture
def image_prompt():
    data = base64.b64encode(b"\x89PNG-not-really-an-image" * 10).decode()
    return (
        Prompt(
            messages=[
                VisionMessage(
                    role=MessageRole.USER,
                    content=[
                        VisionMessageTextContent(text="What is this?"),
                        VisionMessageImageContent(image_url=VisionMessageImageURL(url=f"data:image/png;base64,{data}")),
                        VisionMessageImageContent(image_url=VisionMessageImageURL(url="https://example.com/cat.png")),
                    ],
                )
            ]
        ),
        data,
    )


def test_inline_image_is_replaced_by_a_placeholder_and_a_url_is_kept(mock_llm_executor, config, trace, image_prompt):
    prompt, data = image_prompt
    node = llms.OpenAI(connection=connections.OpenAI(api_key="test-key"), model="gpt-4o", prompt=prompt)

    Workflow(flow=Flow(nodes=[node])).run(input_data={}, config=config)

    span = trace().one("chat gpt-4o")
    assert data not in " ".join(str(v) for v in span.attributes.values())
    (message,) = attrs_json(span, "gen_ai.input.messages")
    text, blob, uri = message["parts"]
    assert text == {"type": "text", "content": "What is this?"}
    assert blob["type"] == "blob"
    assert blob["modality"] == "image"
    assert blob["mime_type"] == "image/png"
    assert blob["content"] == ""
    assert blob["size"] == len(data)
    assert uri == {"type": "uri", "modality": "image", "uri": "https://example.com/cat.png"}


def test_inline_image_is_recorded_when_binary_capture_is_on(mock_llm_executor, make_config, trace, image_prompt):
    prompt, data = image_prompt
    node = llms.OpenAI(connection=connections.OpenAI(api_key="test-key"), model="gpt-4o", prompt=prompt)

    Workflow(flow=Flow(nodes=[node])).run(input_data={}, config=make_config(binary_content_enabled=True))

    (message,) = attrs_json(trace().one("chat gpt-4o"), "gen_ai.input.messages")
    blob = message["parts"][1]
    assert blob["type"] == "blob"
    assert blob["content"] == data
