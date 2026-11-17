import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from litellm import APIError, BadRequestError

from dynamiq import Workflow, connections
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.components.embedders.ollama import OllamaEmbedder
from dynamiq.flows import Flow
from dynamiq.nodes.embedders import OllamaDocumentEmbedder, OllamaTextEmbedder
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types import Document
from dynamiq.utils.litellm import get_litellm
from tests.integration.nodes.embedders.conftest import (
    assert_embedder_failure,
    assert_embedder_success,
    create_document_embedder_workflow,
    create_text_embedder_workflow,
)

OLLAMA_URL = "http://ollama.internal:11434"


@pytest.fixture
def ollama_connection():
    return connections.Ollama(id=str(uuid.uuid4()), url=OLLAMA_URL)


@pytest.fixture
def ollama_model():
    return "ollama/nomic-embed-text"


@pytest.fixture
def ollama_text_embedder(ollama_connection, ollama_model):
    return OllamaTextEmbedder(
        id="text_embedder", name="OllamaTextEmbedder", connection=ollama_connection, model=ollama_model
    )


@pytest.fixture
def ollama_document_embedder(ollama_connection, ollama_model):
    return OllamaDocumentEmbedder(
        id="document_embedder", name="OllamaDocumentEmbedder", connection=ollama_connection, model=ollama_model
    )


@pytest.fixture
def ollama_text_embedder_workflow(ollama_text_embedder):
    return create_text_embedder_workflow(ollama_text_embedder)


@pytest.fixture
def ollama_document_embedder_workflow(ollama_document_embedder):
    return create_document_embedder_workflow(ollama_document_embedder)


def ollama_server(*embed_bodies):
    """Answer litellm's HTTP calls the way an Ollama server does, one /api/embed body per call.

    litellm also looks up model details at /api/show for its cost calculation; that lookup fails as it does when no
    server runs at the default address, which litellm tolerates.
    """
    bodies = iter(embed_bodies)

    def post(url, json, **kwargs):
        if not url.endswith("/api/embed"):
            raise httpx.ConnectError("Connection refused")
        response = MagicMock()
        response.json.return_value = next(bodies)
        return response

    return post


def embed_calls(mock_post):
    return [call.kwargs for call in mock_post.call_args_list if call.kwargs["url"].endswith("/api/embed")]


def test_workflow_with_ollama_text_embedder(
    mock_embedding_executor, ollama_text_embedder_workflow, query_input, ollama_model
):
    workflow, embedder, output_node = ollama_text_embedder_workflow

    response = workflow.run(
        input_data=query_input,
        config=RunnableConfig(callbacks=[TracingCallbackHandler()]),
    )

    assert_embedder_success(response, embedder, output_node)
    mock_embedding_executor.assert_called_once_with(
        input=[query_input["query"]],
        model=ollama_model,
        api_base=OLLAMA_URL,
    )


def test_workflow_with_ollama_document_embedder(
    mock_embedding_executor, ollama_document_embedder_workflow, document_input, ollama_model
):
    workflow, embedder, output_node = ollama_document_embedder_workflow

    response = workflow.run(
        input_data=document_input,
        config=RunnableConfig(callbacks=[TracingCallbackHandler()]),
    )

    assert_embedder_success(response, embedder, output_node)
    mock_embedding_executor.assert_called_once_with(
        input=[document_input["documents"][0].content],
        model=ollama_model,
        api_base=OLLAMA_URL,
    )


@pytest.mark.parametrize("node_cls", [OllamaTextEmbedder, OllamaDocumentEmbedder])
def test_default_connection_points_to_local_server(node_cls):
    node = node_cls()

    component = node.text_embedder if node_cls is OllamaTextEmbedder else node.document_embedder

    assert component.model == "ollama/nomic-embed-text"
    assert component.embed_params == {"api_base": "http://localhost:11434"}


@pytest.mark.parametrize("node_cls", [OllamaTextEmbedder, OllamaDocumentEmbedder])
def test_model_without_prefix_is_routed_to_ollama(ollama_connection, node_cls):
    node = node_cls(connection=ollama_connection, model="mxbai-embed-large")

    component = node.text_embedder if node_cls is OllamaTextEmbedder else node.document_embedder

    assert component.model == "ollama/mxbai-embed-large"


def test_server_url_is_sent_although_the_connection_has_a_client(ollama_text_embedder):
    # The Ollama connection's client is the requests module; sending it instead of the URL would reach localhost.
    assert ollama_text_embedder.client is not None

    assert ollama_text_embedder.text_embedder.embed_params == {"api_base": OLLAMA_URL}
    assert ollama_text_embedder.text_embedder.embed_params_async == {"api_base": OLLAMA_URL}


def test_text_embedder_calls_ollama_embed_endpoint(ollama_text_embedder_workflow, query_input):
    workflow, embedder, output_node = ollama_text_embedder_workflow
    litellm = get_litellm()

    with patch.object(
        litellm.module_level_client,
        "post",
        side_effect=ollama_server({"embeddings": [[0.1, 0.2, 0.3]], "prompt_eval_count": 5}),
    ) as mock_post:
        response = workflow.run(input_data=query_input)

    assert_embedder_success(response, embedder, output_node, expected_embedding_length=3)
    assert embed_calls(mock_post) == [
        {"url": f"{OLLAMA_URL}/api/embed", "json": {"model": "nomic-embed-text", "input": [query_input["query"]]}}
    ]
    assert response.output[embedder.id]["output"]["embedding"] == [0.1, 0.2, 0.3]


def test_document_embedder_sends_each_batch_in_one_request(ollama_connection, ollama_model):
    embedder = OllamaEmbedder(connection=ollama_connection, model=ollama_model, batch_size=2)
    documents = [Document(content=f"document {index}") for index in range(3)]
    litellm = get_litellm()

    with patch.object(
        litellm.module_level_client,
        "post",
        side_effect=ollama_server(
            {"embeddings": [[1.0], [2.0]], "prompt_eval_count": 4},
            {"embeddings": [[3.0]], "prompt_eval_count": 2},
        ),
    ) as mock_post:
        result = embedder.embed_documents(documents)

    assert [call["json"]["input"] for call in embed_calls(mock_post)] == [
        ["document 0", "document 1"],
        ["document 2"],
    ]
    assert [document.embedding for document in result["documents"]] == [[1.0], [2.0], [3.0]]
    assert result["meta"]["model"] == "ollama/nomic-embed-text"
    assert result["meta"]["usage"]["prompt_tokens"] == 6
    assert result["meta"]["usage"]["total_tokens"] == 6


@pytest.mark.asyncio
async def test_text_embedder_async_calls_ollama_embed_endpoint(ollama_text_embedder, query_input):
    litellm = get_litellm()

    with (
        patch.object(
            litellm.module_level_aclient,
            "post",
            new_callable=AsyncMock,
            side_effect=ollama_server({"embeddings": [[0.4, 0.5]], "prompt_eval_count": 3}),
        ) as mock_post,
        patch.object(litellm.module_level_client, "post", side_effect=ollama_server()),
    ):
        result = await ollama_text_embedder.run_async(input_data=query_input)

    assert result.status == RunnableStatus.SUCCESS
    assert result.output["embedding"] == [0.4, 0.5]
    assert embed_calls(mock_post) == [
        {"url": f"{OLLAMA_URL}/api/embed", "json": {"model": "nomic-embed-text", "input": [query_input["query"]]}}
    ]


@pytest.mark.asyncio
async def test_workflow_with_ollama_text_embedder_async(
    mock_aembedding_executor, ollama_text_embedder, query_input, ollama_model
):
    result = await ollama_text_embedder.run_async(input_data=query_input)

    assert result.status == RunnableStatus.SUCCESS
    mock_aembedding_executor.assert_awaited_once_with(
        input=[query_input["query"]],
        model=ollama_model,
        api_base=OLLAMA_URL,
    )


@pytest.mark.asyncio
async def test_workflow_with_ollama_document_embedder_async(
    mock_aembedding_executor, ollama_document_embedder, document_input, ollama_model
):
    result = await ollama_document_embedder.run_async(input_data=document_input)

    assert result.status == RunnableStatus.SUCCESS
    mock_aembedding_executor.assert_awaited_once_with(
        input=[document_input["documents"][0].content],
        model=ollama_model,
        api_base=OLLAMA_URL,
    )


@pytest.mark.parametrize(
    "error",
    [
        APIError(500, "Service unavailable", "ollama", "ollama/nomic-embed-text"),
        BadRequestError('model "nomic-embed-text" not found, try pulling it first', "nomic-embed-text", "ollama"),
    ],
)
def test_text_embedder_api_errors(ollama_text_embedder_workflow, query_input, error):
    workflow, embedder, output_node = ollama_text_embedder_workflow

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding", side_effect=error):
        response = workflow.run(input_data=query_input)

    assert_embedder_failure(response, embedder, output_node, type(error).__name__, error.message)


def test_text_embedder_empty_input(ollama_text_embedder_workflow, empty_query_input):
    workflow, embedder, output_node = ollama_text_embedder_workflow
    response = workflow.run(input_data=empty_query_input)
    assert_embedder_failure(response, embedder, output_node)


def test_document_embedder_empty_content(ollama_document_embedder_workflow, empty_document_content_input):
    workflow, embedder, output_node = ollama_document_embedder_workflow
    response = workflow.run(input_data=empty_document_content_input)
    assert_embedder_failure(response, embedder, output_node)


def test_document_embedder_api_returns_empty_embedding(
    ollama_document_embedder_workflow, document_input, empty_embedding_response_factory, ollama_model
):
    workflow, embedder, output_node = ollama_document_embedder_workflow

    with patch(
        "dynamiq.components.embedders.base.BaseEmbedder._embedding",
        return_value=empty_embedding_response_factory(ollama_model),
    ):
        response = workflow.run(input_data=document_input)

    assert_embedder_failure(response, embedder, output_node, expected_error_message="Embedding is empty")


def test_ollama_embedders_yaml_roundtrip(tmp_path, ollama_connection):
    text_embedder = OllamaTextEmbedder(id="text-embedder", connection=ollama_connection, model="mxbai-embed-large")
    document_embedder = OllamaDocumentEmbedder(id="document-embedder", connection=ollama_connection)
    workflow = Workflow(id="ollama-workflow", flow=Flow(id="ollama-flow", nodes=[text_embedder, document_embedder]))

    yaml_path = tmp_path / "ollama_workflow.yaml"
    workflow.to_yaml_file(str(yaml_path))
    loaded = Workflow.from_yaml_file(str(yaml_path), init_components=True)

    roundtrip_path = tmp_path / "ollama_workflow_roundtrip.yaml"
    loaded.to_yaml_file(str(roundtrip_path))
    roundtrip = Workflow.from_yaml_file(str(roundtrip_path), init_components=True)

    nodes = {node.id: node for node in roundtrip.flow.nodes}
    assert isinstance(nodes["text-embedder"], OllamaTextEmbedder)
    assert isinstance(nodes["document-embedder"], OllamaDocumentEmbedder)
    assert nodes["text-embedder"].connection.url == OLLAMA_URL
    assert nodes["text-embedder"].text_embedder.model == "ollama/mxbai-embed-large"
    assert nodes["text-embedder"].text_embedder.embed_params == {"api_base": OLLAMA_URL}
    assert nodes["document-embedder"].document_embedder.model == "ollama/nomic-embed-text"
