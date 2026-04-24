from unittest.mock import AsyncMock, patch

import pytest
from litellm.types.utils import EmbeddingResponse, Usage

from dynamiq.components.embedders.openai import OpenAIEmbedder
from dynamiq.connections import OpenAI as OpenAIConnection


def _make_embedding_response(model: str, embedding: list[float]) -> EmbeddingResponse:
    response = EmbeddingResponse()
    response["data"] = [{"embedding": embedding}]
    response["model"] = model
    response["usage"] = Usage(prompt_tokens=1, completion_tokens=0, total_tokens=1)
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


class TestEmbedTextsBatchAsync:
    @pytest.mark.asyncio
    async def test_batch_async_splits_inputs_by_batch_size(self):
        embedder = _make_openai_embedder()
        embedder.batch_size = 2

        async def fake_aembedding(model, input, **kwargs):
            response = EmbeddingResponse()
            response["data"] = [{"embedding": [float(i)]} for i in range(len(input))]
            response["model"] = model
            response["usage"] = Usage(
                prompt_tokens=len(input),
                completion_tokens=0,
                total_tokens=len(input),
            )
            return response

        embedder._aembedding = AsyncMock(side_effect=fake_aembedding)

        texts = ["a", "b", "c", "d", "e"]
        embeddings, meta = await embedder._embed_texts_batch_async(
            texts_to_embed=texts, batch_size=2
        )

        # 5 texts, batch_size=2 → 3 calls.
        assert embedder._aembedding.await_count == 3
        assert len(embeddings) == 5
        assert meta["model"] == "text-embedding-3-small"
        assert meta["usage"]["total_tokens"] == 5


class TestEmbedDocumentsAsync:
    @pytest.mark.asyncio
    async def test_embed_documents_async_populates_embeddings(self):
        from dynamiq.types import Document

        embedder = _make_openai_embedder()

        async def fake_aembedding(model, input, **kwargs):
            response = EmbeddingResponse()
            response["data"] = [{"embedding": [float(i)]} for i in range(len(input))]
            response["model"] = model
            response["usage"] = Usage(
                prompt_tokens=len(input),
                completion_tokens=0,
                total_tokens=len(input),
            )
            return response

        embedder._aembedding = AsyncMock(side_effect=fake_aembedding)

        docs = [Document(content="one"), Document(content="two")]
        result = await embedder.embed_documents_async(docs)

        assert result["documents"] is docs
        assert docs[0].embedding == [0.0]
        assert docs[1].embedding == [1.0]
        assert result["meta"]["model"] == "text-embedding-3-small"

    @pytest.mark.asyncio
    async def test_embed_documents_async_empty_returns_early(self):
        embedder = _make_openai_embedder()
        embedder._aembedding = AsyncMock()

        result = await embedder.embed_documents_async([])

        assert result == {"documents": [], "meta": {}}
        embedder._aembedding.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_embed_documents_async_rejects_non_list(self):
        embedder = _make_openai_embedder()
        with pytest.raises(TypeError, match="DocumentEmbedder expects a list"):
            await embedder.embed_documents_async("not a list")


class TestHuggingFaceAsyncOverrides:
    @pytest.mark.asyncio
    async def test_hf_embed_text_async_sends_single_input_not_list(self):
        from dynamiq.components.embedders.huggingface import HuggingFaceEmbedder
        from dynamiq.connections import HuggingFace as HuggingFaceConnection

        embedder = HuggingFaceEmbedder(
            connection=HuggingFaceConnection(api_key="hf-test"),
        )
        embedder._aembedding = AsyncMock(
            return_value=_make_embedding_response(
                model=embedder.model, embedding=[0.1, 0.2]
            )
        )

        result = await embedder.embed_text_async("hello")

        call_kwargs = embedder._aembedding.await_args.kwargs
        # HF sends a single string, not a list.
        assert call_kwargs["input"] == "hello"
        assert "api_base" in call_kwargs
        assert result["embedding"] == [0.1, 0.2]

    @pytest.mark.asyncio
    async def test_hf_batch_async_iterates_one_by_one(self):
        from dynamiq.components.embedders.huggingface import HuggingFaceEmbedder
        from dynamiq.connections import HuggingFace as HuggingFaceConnection

        embedder = HuggingFaceEmbedder(
            connection=HuggingFaceConnection(api_key="hf-test"),
        )

        async def fake_aembedding(model, input, **kwargs):
            response = EmbeddingResponse()
            response["data"] = [{"embedding": [0.0]}]
            response["model"] = model
            response["usage"] = Usage(
                prompt_tokens=1,
                completion_tokens=0,
                total_tokens=1,
            )
            return response

        embedder._aembedding = AsyncMock(side_effect=fake_aembedding)

        texts = ["a", "b", "c"]
        embeddings, meta = await embedder._embed_texts_batch_async(
            texts_to_embed=texts, batch_size=2
        )

        # Non-batched API → one call per text.
        assert embedder._aembedding.await_count == 3
        # Each call received a single string, not a list.
        for call in embedder._aembedding.await_args_list:
            assert isinstance(call.kwargs["input"], str)
        assert len(embeddings) == 3
        assert meta["usage"]["total_tokens"] == 3
