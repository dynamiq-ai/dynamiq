import pytest

from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.nodes.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.runnables import RunnableStatus
from dynamiq.types.document import Document


@pytest.mark.asyncio
async def test_openai_text_embedder_run_async():
    """Single async embed call must return a real embedding vector."""
    node = OpenAITextEmbedder(connection=OpenAIConnection())
    result = await node.run_async(input_data={"query": "hello world"})

    assert result.status == RunnableStatus.SUCCESS, f"failed: {result.output}"
    embedding = result.output["embedding"]
    assert isinstance(embedding, list) and len(embedding) > 0
    assert all(isinstance(x, float) for x in embedding)


@pytest.mark.asyncio
async def test_openai_document_embedder_run_async():
    """Async embed across multiple documents."""
    node = OpenAIDocumentEmbedder(connection=OpenAIConnection())
    docs = [Document(content="alpha"), Document(content="beta"), Document(content="gamma")]
    result = await node.run_async(input_data={"documents": docs})

    assert result.status == RunnableStatus.SUCCESS, f"failed: {result.output}"
    embedded = result.output["documents"]
    assert len(embedded) == 3
    for d in embedded:
        assert d.embedding is not None
        assert len(d.embedding) > 0


@pytest.mark.asyncio
async def test_openai_text_embedder_workflow_run_async():
    """Full Workflow.run_async lifecycle."""
    node = OpenAITextEmbedder(connection=OpenAIConnection())
    workflow = Workflow(flow=Flow(nodes=[node]))

    result = await workflow.run_async(input_data={"query": "the quick brown fox"})
    assert result.status == RunnableStatus.SUCCESS, f"workflow failed: {result.output}"

    node_output = result.output[node.id]
    assert node_output["status"] == "success"
    assert len(node_output["output"]["embedding"]) > 0


def test_openai_embedder_async_params_omit_sync_client():
    """``embed_params_async`` on the underlying component must NOT
    include the sync OpenAI client cached by ``ConnectionNode.init_components``.
    """
    from openai import OpenAI as OpenAIClient

    node = OpenAITextEmbedder(connection=OpenAIConnection())
    # init_components caches a sync OpenAI client on the underlying component
    assert isinstance(node.text_embedder.client, OpenAIClient)
    # and the sync params correctly include it
    assert node.text_embedder.embed_params.get("client") is node.text_embedder.client

    # but the async params must drop it and fall back to api_key/api_base only
    params = node.text_embedder.embed_params_async
    assert "client" not in params, "async embed params must not pass the sync client"
    assert "api_key" in params and "api_base" in params


def test_openai_embedder_async_params_preserve_dimensions():
    """``embed_params_async`` must preserve subclass-specific params like ``dimensions``.

    OpenAIEmbedderComponent overrides ``embed_params`` to add ``dimensions`` when set.
    A naive async params builder that only returned ``conn_params`` would silently drop
    this and produce embeddings of the model's default size instead of the requested one.
    """
    node = OpenAITextEmbedder(connection=OpenAIConnection(), dimensions=512)

    sync_params = node.text_embedder.embed_params
    async_params = node.text_embedder.embed_params_async

    assert sync_params.get("dimensions") == 512
    assert async_params.get("dimensions") == 512
    assert "client" not in async_params
    assert "api_key" in async_params and "api_base" in async_params


@pytest.mark.asyncio
async def test_openai_text_embedder_dimensions_round_trip_async():
    """End-to-end: setting ``dimensions`` on the async path must produce vectors of that size."""
    node = OpenAITextEmbedder(connection=OpenAIConnection(), dimensions=512)
    result = await node.run_async(input_data={"query": "hello world"})

    assert result.status == RunnableStatus.SUCCESS, f"failed: {result.output}"
    embedding = result.output["embedding"]
    assert len(embedding) == 512, (
        f"expected 512-d vector via async path; got {len(embedding)}. "
        f"Subclass-specific dimensions param was likely dropped from embed_params_async."
    )
