from unittest.mock import AsyncMock, MagicMock

import pytest
from litellm.types.utils import EmbeddingResponse, Usage

from dynamiq.components.embedders.huggingface import HuggingFaceEmbedder
from dynamiq.connections import HuggingFace as HuggingFaceConnection


def _make_embedding_response() -> EmbeddingResponse:
    response = EmbeddingResponse()
    response["data"] = [{"embedding": [0.1, 0.2]}]
    response["model"] = "huggingface/x"
    response["usage"] = Usage(prompt_tokens=1, completion_tokens=0, total_tokens=1)
    return response


def _make_embedder() -> HuggingFaceEmbedder:
    embedder = HuggingFaceEmbedder(
        model="huggingface/x", connection=HuggingFaceConnection(api_key="test-key"), client=MagicMock()
    )
    embedder._aembedding = AsyncMock(return_value=_make_embedding_response())
    return embedder


def _assert_async_params(kwargs: dict):
    assert "client" not in kwargs
    assert kwargs["api_key"] == "test-key"
    assert kwargs["api_base"] == "https://api-inference.huggingface.co/models/x"


@pytest.mark.asyncio
async def test_embed_text_async_drops_client_and_keeps_api_key():
    embedder = _make_embedder()
    await embedder.embed_text_async("hi")
    _assert_async_params(embedder._aembedding.await_args.kwargs)


@pytest.mark.asyncio
async def test_embed_texts_batch_async_drops_client_and_keeps_api_key():
    embedder = _make_embedder()
    await embedder._embed_texts_batch_async(["hi"], batch_size=1)
    _assert_async_params(embedder._aembedding.await_args.kwargs)
