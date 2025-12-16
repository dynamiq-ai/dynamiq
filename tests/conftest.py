import io

import pytest
from fakeredis import FakeRedis
from litellm import ModelResponse
from litellm.types.utils import EmbeddingResponse
from litellm.utils import Delta

from dynamiq import connections, prompts
from dynamiq.cache.backends import RedisCache
from dynamiq.clients import BaseTracingClient
from dynamiq.nodes import llms
from dynamiq.types.document import Document


@pytest.fixture(autouse=True)
def autouse_fixture(
    mock_tracing_client,
    mock_redis_backend,
): ...


@pytest.fixture
def mock_llm_response_text():
    return "mocked_response"


@pytest.fixture
def mock_whisper_response_text():
    return "Welcome."


@pytest.fixture
def mock_elevenlabs_response_text():
    return b"Mock text"


@pytest.fixture
def mock_redis():
    return FakeRedis()


@pytest.fixture
def mock_llm_executor(mocker, mock_llm_response_text):
    def mock_completion_streaming_obj(mock_response):
        for chunk in mock_response:
            model_r = ModelResponse(stream=True)
            model_r.choices[0].delta = Delta(**{"role": "assistant", "content": chunk})
            yield model_r

    def response(stream: bool, *args, **kwargs):
        if stream:
            return mock_completion_streaming_obj(mock_response=mock_llm_response_text)

        model_r = ModelResponse()
        model_r["choices"][0]["message"]["content"] = mock_llm_response_text
        return model_r

    mock_llm = mocker.patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=response)
    yield mock_llm


@pytest.fixture
def mock_embedding_executor(mocker):
    def response(*args, **kwargs):
        embed_r = EmbeddingResponse()
        embedding_data = {"embedding": [0]}
        embed_r["data"] = [embedding_data]
        embed_r["model"] = kwargs.get("model")
        embed_r["usage"] = {"usage": {"prompt_tokens": 6, "completion_tokens": 0, "total_tokens": 6}}
        return embed_r

    mock_llm = mocker.patch(
        "dynamiq.components.embedders.base.BaseEmbedder._embedding",
        side_effect=response,
    )
    yield mock_llm


@pytest.fixture
def mock_embedding_tracing_output():
    return {
        "embedding": [
            0.12,
            -0.87,
            0.45,
            0.98,
            -0.34,
            0.67,
            -0.23,
            0.11,
            -0.56,
            0.89,
            -0.32,
            0.76,
            -0.12,
            0.44,
            0.92,
            -0.68,
            0.15,
            0.37,
            -0.45,
            0.88,
            -0.14,
            0.59,
            -0.72,
            0.25,
            0.77,
            -0.91,
            0.63,
            -0.35,
            0.41,
            0.99,
        ]
    }


@pytest.fixture
def mock_embedding_executor_truncate_tracing(mocker, mock_embedding_tracing_output):
    def response(*args, **kwargs):
        embed_r = EmbeddingResponse()
        embed_r["data"] = [mock_embedding_tracing_output]
        embed_r["model"] = kwargs.get("model")
        embed_r["usage"] = {"usage": {"prompt_tokens": 6, "completion_tokens": 0, "total_tokens": 6}}
        return embed_r

    mock_llm = mocker.patch(
        "dynamiq.components.embedders.base.BaseEmbedder._embedding",
        side_effect=response,
    )
    yield mock_llm


@pytest.fixture
def mock_tracing_client(mocker):
    class MockTracingClient(BaseTracingClient):
        def trace(self, runs) -> None:
            pass

    mocker.patch.object(MockTracingClient, "trace")
    yield MockTracingClient


@pytest.fixture
def mock_redis_backend(mocker, mock_redis):
    yield mocker.patch(
        "dynamiq.cache.backends.RedisCache.from_config",
        return_value=RedisCache(client=mock_redis),
    )


@pytest.fixture()
def ai_prompt():
    return prompts.Prompt(
        messages=[
            prompts.Message(
                role="user",
                content="What is AI?",
            ),
        ],
    )


@pytest.fixture()
def ds_prompt():
    return prompts.Prompt(
        messages=[
            prompts.Message(
                role="user",
                content="What is DS?",
            ),
        ],
    )


@pytest.fixture()
def openai_node(ai_prompt):
    return llms.OpenAI(
        name="OpenAI",
        model="gpt-3.5-turbo",
        connection=connections.OpenAI(
            api_key="test-api-key",
        ),
        prompt=ai_prompt,
        is_postponed_component_init=True,
    )


@pytest.fixture()
def anthropic_node(ds_prompt):
    return llms.Anthropic(
        name="Anthropic",
        model="claude-3-opus-20240229",
        connection=connections.Anthropic(
            api_key="test-api-key",
        ),
        prompt=ds_prompt,
        is_postponed_component_init=True,
    )


@pytest.fixture
def mock_documents():
    return [
        Document(id="1", content="Document 1", embedding=[0.1, 0.1, 0.2], metadata={"file_id": "file_id_1"}),
        Document(id="2", content="Document 2", embedding=[0.1, 0.1, 0.2], metadata={"file_id": "file_id_1"}),
    ]


@pytest.fixture()
def mock_filters():
    return {
        "operator": "AND",
        "conditions": [
            {"field": "years", "operator": "==", "value": "2019"},
            {"field": "companies", "operator": "in", "value": ["BMW", "Mercedes"]},
        ],
    }


@pytest.fixture
def mock_image_url():
    return "https://example.com/generated_image.png"


@pytest.fixture
def mock_image_b64():
    return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="


@pytest.fixture
def mock_image_bytes(mock_image_b64):
    import base64

    return base64.b64decode(mock_image_b64)


@pytest.fixture
def mock_image_generation_executor(mocker, mock_image_url, mock_image_b64):
    """Mock for litellm.image_generation with URL and b64_json support."""

    def response(*args, **kwargs):
        from types import SimpleNamespace

        response_format = kwargs.get("response_format", "url")
        n = kwargs.get("n") or 1  # Handle None explicitly

        data = []
        for _ in range(n):
            if response_format == "b64_json":
                img_data = SimpleNamespace(b64_json=mock_image_b64, url=None)
            else:
                img_data = SimpleNamespace(url=mock_image_url, b64_json=None)
            data.append(img_data)

        return SimpleNamespace(
            data=data,
            created=1234567890,
        )

    mock_gen = mocker.patch(
        "dynamiq.nodes.images.generation.ImageGeneration._image_generation",
        side_effect=response,
    )
    yield mock_gen


@pytest.fixture
def mock_image_edit_executor(mocker, mock_image_url, mock_image_b64):
    """Mock for litellm.image_edit with URL and b64_json support."""

    def response(*args, **kwargs):
        from types import SimpleNamespace

        response_format = kwargs.get("response_format", "url")
        n = kwargs.get("n") or 1

        data = []
        for _ in range(n):
            if response_format == "b64_json":
                img_data = SimpleNamespace(b64_json=mock_image_b64, url=None)
            else:
                img_data = SimpleNamespace(url=mock_image_url, b64_json=None)
            data.append(img_data)

        return SimpleNamespace(
            data=data,
            created=1234567890,
        )

    mock_edit = mocker.patch(
        "dynamiq.nodes.images.edit.ImageEdit._image_edit",
        side_effect=response,
    )
    yield mock_edit


@pytest.fixture
def mock_image_variation_executor(mocker, mock_image_url, mock_image_b64):
    """Mock for litellm.image_variation with URL and b64_json support."""

    def response(*args, **kwargs):
        from types import SimpleNamespace

        response_format = kwargs.get("response_format", "url")
        n = kwargs.get("n") or 1

        data = []
        for _ in range(n):
            if response_format == "b64_json":
                img_data = SimpleNamespace(b64_json=mock_image_b64, url=None)
            else:
                img_data = SimpleNamespace(url=mock_image_url, b64_json=None)
            data.append(img_data)

        return SimpleNamespace(
            data=data,
            created=1234567890,
        )

    mock_var = mocker.patch(
        "dynamiq.nodes.images.variation.ImageVariation._image_variation",
        side_effect=response,
    )
    yield mock_var


@pytest.fixture
def mock_httpx_get(mocker, mock_image_bytes):
    """Mock httpx.get for downloading images from URLs."""
    from types import SimpleNamespace

    def response(*args, **kwargs):
        return SimpleNamespace(
            content=mock_image_bytes,
            raise_for_status=lambda: None,
        )

    mock_get = mocker.patch("httpx.get", side_effect=response)
    yield mock_get


@pytest.fixture
def mock_image_file(mock_image_bytes):
    """Create a test image file."""
    image_file = io.BytesIO(mock_image_bytes)
    image_file.name = "test_image.png"
    image_file.seek(0)
    return image_file
