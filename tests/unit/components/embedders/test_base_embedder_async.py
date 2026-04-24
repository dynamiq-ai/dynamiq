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
