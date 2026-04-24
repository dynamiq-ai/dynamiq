from unittest.mock import AsyncMock, patch

import pytest
from litellm.types.utils import EmbeddingResponse

from dynamiq.components.embedders.openai import OpenAIEmbedder
from dynamiq.connections import OpenAI as OpenAIConnection


def _make_embedding_response(model: str, embedding: list[float]) -> EmbeddingResponse:
    response = EmbeddingResponse()
    response["data"] = [{"embedding": embedding}]
    response["model"] = model
    response["usage"] = {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1}
    return response


def _make_openai_embedder() -> OpenAIEmbedder:
    return OpenAIEmbedder(
        connection=OpenAIConnection(api_key="test-key"),
        model="text-embedding-3-small",
    )


class TestBaseEmbedderAembeddingBound:
    def test_aembedding_is_bound(self):
        """BaseEmbedder should expose _aembedding after __init__."""
        embedder = _make_openai_embedder()
        assert embedder._aembedding is not None
        assert callable(embedder._aembedding)


class TestEmbedTextAsync:
    @pytest.mark.asyncio
    async def test_embed_text_async_calls_aembedding_with_single_input(self):
        embedder = _make_openai_embedder()
        embedder._aembedding = AsyncMock(
            return_value=_make_embedding_response(
                model="text-embedding-3-small", embedding=[0.1, 0.2, 0.3]
            )
        )

        result = await embedder.embed_text_async("hello world")

        embedder._aembedding.assert_awaited_once()
        call_kwargs = embedder._aembedding.await_args.kwargs
        assert call_kwargs["model"] == "text-embedding-3-small"
        assert call_kwargs["input"] == ["hello world"]
        assert result["embedding"] == [0.1, 0.2, 0.3]
        assert result["meta"]["model"] == "text-embedding-3-small"

    @pytest.mark.asyncio
    async def test_embed_text_async_rejects_non_string(self):
        embedder = _make_openai_embedder()
        with pytest.raises(TypeError, match="TextEmbedder expects a string"):
            await embedder.embed_text_async(["not", "a", "string"])

    @pytest.mark.asyncio
    async def test_embed_text_async_raises_on_invalid_embedding(self):
        embedder = _make_openai_embedder()
        embedder._aembedding = AsyncMock(
            return_value=_make_embedding_response(
                model="text-embedding-3-small", embedding=[]
            )
        )
        with pytest.raises(ValueError, match="Invalid embedding"):
            await embedder.embed_text_async("hello")
