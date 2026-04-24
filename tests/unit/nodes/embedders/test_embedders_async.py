import asyncio
from unittest.mock import AsyncMock

import pytest
from litellm.types.utils import EmbeddingResponse, Usage

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.embedders.openai import OpenAIDocumentEmbedder, OpenAITextEmbedder
from dynamiq.types import Document
from tests.helpers.async_node import assert_concurrent_execution


def _make_response(model, embedding):
    response = EmbeddingResponse()
    response["data"] = [{"embedding": embedding}]
    response["model"] = model
    response["usage"] = Usage(prompt_tokens=1, completion_tokens=0, total_tokens=1)
    return response


class TestEmbedderNodesHaveNativeAsync:
    def test_text_embedder_reports_native_async(self):
        node = OpenAITextEmbedder(
            name="t", connection=OpenAIConnection(api_key="k"), model="text-embedding-3-small"
        )
        assert node.has_native_async is True

    def test_document_embedder_reports_native_async(self):
        node = OpenAIDocumentEmbedder(
            name="d", connection=OpenAIConnection(api_key="k"), model="text-embedding-3-small"
        )
        assert node.has_native_async is True


class TestTextEmbedderExecuteAsync:
    @pytest.mark.asyncio
    async def test_execute_async_calls_component_async_path(self):
        node = OpenAITextEmbedder(
            name="t", connection=OpenAIConnection(api_key="k"), model="text-embedding-3-small"
        )
        node.text_embedder._aembedding = AsyncMock(
            return_value=_make_response("text-embedding-3-small", [0.1, 0.2])
        )

        result = await node.run_async(input_data={"query": "hello"})

        assert result.output["embedding"] == [0.1, 0.2]
        assert result.output["query"] == "hello"
        node.text_embedder._aembedding.assert_awaited_once()


class TestDocumentEmbedderExecuteAsync:
    @pytest.mark.asyncio
    async def test_execute_async_populates_document_embeddings(self):
        node = OpenAIDocumentEmbedder(
            name="d", connection=OpenAIConnection(api_key="k"), model="text-embedding-3-small"
        )

        async def fake_aembedding(model, input, **kwargs):
            resp = EmbeddingResponse()
            resp["data"] = [{"embedding": [float(i)]} for i in range(len(input))]
            resp["model"] = model
            resp["usage"] = Usage(
                prompt_tokens=len(input),
                completion_tokens=0,
                total_tokens=len(input),
            )
            return resp

        node.document_embedder._aembedding = AsyncMock(side_effect=fake_aembedding)

        docs = [Document(content="a"), Document(content="b")]
        result = await node.run_async(input_data={"documents": docs})

        assert result.output["documents"][0].embedding == [0.0]
        assert result.output["documents"][1].embedding == [1.0]


class TestEmbedderConcurrency:
    @pytest.mark.asyncio
    async def test_document_embedder_calls_overlap_under_gather(self):
        node = OpenAIDocumentEmbedder(
            name="d", connection=OpenAIConnection(api_key="k"), model="text-embedding-3-small"
        )

        single_call_s = 0.1

        async def slow_aembedding(model, input, **kwargs):
            await asyncio.sleep(single_call_s)
            resp = EmbeddingResponse()
            resp["data"] = [{"embedding": [0.0]} for _ in input]
            resp["model"] = model
            resp["usage"] = Usage(
                prompt_tokens=len(input),
                completion_tokens=0,
                total_tokens=len(input),
            )
            return resp

        node.document_embedder._aembedding = AsyncMock(side_effect=slow_aembedding)

        payloads = [{"documents": [Document(content=f"doc-{i}")]} for i in range(10)]

        results = await assert_concurrent_execution(
            node,
            payloads,
            expected_single_call_s=single_call_s,
            tolerance_factor=3.0,
        )

        assert len(results) == 10
        for r in results:
            assert r.output["documents"][0].embedding == [0.0]
