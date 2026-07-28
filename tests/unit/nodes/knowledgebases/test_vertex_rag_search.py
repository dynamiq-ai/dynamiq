from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from dynamiq import Workflow
from dynamiq.connections import VertexAI
from dynamiq.flows import Flow
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.knowledgebases.vertex import VertexAIRagSearch, VertexAIRagSearchInputSchema
from dynamiq.runnables import RunnableConfig
from dynamiq.types import Document

CORPUS_NAME = "projects/test-project/locations/us-central1/ragCorpora/123"


@pytest.fixture(scope="module")
def service_account_private_key():
    """A real (throwaway) RSA key so google-auth can parse the service-account info offline."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    ).decode()


@pytest.fixture
def connection(service_account_private_key):
    return VertexAI(
        id="vertex-conn",
        project_id="test-project",
        private_key_id="key-id",
        private_key=service_account_private_key,
        client_email="sa@test-project.iam.gserviceaccount.com",
        client_id="1234567890",
        auth_uri="https://accounts.google.com/o/oauth2/auth",
        token_uri="https://oauth2.googleapis.com/token",
        auth_provider_x509_cert_url="https://www.googleapis.com/oauth2/v1/certs",
        client_x509_cert_url="https://www.googleapis.com/robot/v1/metadata/x509/sa",
        universe_domain="googleapis.com",
        vertex_project_id="test-project",
        vertex_project_location="us-central1",
    )


@pytest.fixture
def tool(connection):
    return VertexAIRagSearch(connection=connection, rag_corpus_id="123", client=MagicMock())


def _proto_response(context_specs):
    from google.cloud import aiplatform_v1

    response = aiplatform_v1.RetrieveContextsResponse()
    for spec in context_specs:
        context = aiplatform_v1.RagContexts.Context(
            source_uri=spec.get("source_uri", ""),
            source_display_name=spec.get("source_display_name", ""),
            text=spec.get("text", ""),
        )
        if "score" in spec:
            context.score = spec["score"]
        if "page_first" in spec:
            context.chunk.page_span.first_page = spec["page_first"]
            context.chunk.page_span.last_page = spec.get("page_last", spec["page_first"])
        response.contexts.contexts.append(context)
    return response


def test_type_resolves_to_module_path(tool):
    assert tool.type == "dynamiq.nodes.knowledgebases.VertexAIRagSearch"


def test_query_is_required():
    with pytest.raises(ValidationError):
        VertexAIRagSearchInputSchema()


def test_all_inputs_are_visible_to_agent():
    from dynamiq.nodes.agents.components.schema_generator import generate_function_calling_schemas

    tool_name = "vertexai-rag-search"
    tool = MagicMock()
    tool.name = tool_name
    tool.description = "test"
    tool.input_schema = VertexAIRagSearchInputSchema
    tool.resolved_input_schema = VertexAIRagSearchInputSchema

    schema = next(
        s
        for s in generate_function_calling_schemas(
            tools=[tool], delegation_allowed=False, sanitize_tool_name=lambda name: name
        )
        if s["function"]["name"] == tool_name
    )
    properties = schema["function"]["parameters"]["properties"]

    assert "query" in properties
    assert "top_k" in properties
    assert "metadata_filter" in properties


def test_execute_builds_request_and_parses_documents(tool):
    tool.client.retrieve_contexts.return_value = _proto_response(
        [
            {
                "source_uri": "gs://bucket/handbook.pdf",
                "source_display_name": "handbook.pdf",
                "text": "Employees get 25 vacation days.",
                "score": 0.42,
                "page_first": 3,
                "page_last": 4,
            },
            {"source_uri": "gs://bucket/faq.md", "text": "Vacation carries over.", "score": 0.61},
        ]
    )

    result = tool.execute(VertexAIRagSearchInputSchema(query="vacation days"), RunnableConfig(callbacks=[]))

    call_kwargs = tool.client.retrieve_contexts.call_args.kwargs
    assert call_kwargs["timeout"] == 30
    request = call_kwargs["request"]
    assert request.parent == "projects/test-project/locations/us-central1"
    assert request.query.text == "vacation days"
    assert request.query.rag_retrieval_config.top_k == 10  # node default
    assert request.vertex_rag_store.rag_resources[0].rag_corpus == CORPUS_NAME
    assert list(request.vertex_rag_store.rag_resources[0].rag_file_ids) == []

    documents = result["documents"]
    assert all(isinstance(doc, Document) for doc in documents)
    assert documents[0].content == "Employees get 25 vacation days."
    assert documents[0].score == pytest.approx(0.42)
    assert documents[0].metadata["source_uri"] == "gs://bucket/handbook.pdf"
    assert documents[0].metadata["source_display_name"] == "handbook.pdf"
    assert documents[0].metadata["page_first"] == 3
    assert documents[0].metadata["page_last"] == 4
    assert documents[0].metadata["rag_corpus"] == CORPUS_NAME
    assert "--- Retrieved Source 1 ---" in result["content"]
    assert "source_uri: gs://bucket/handbook.pdf" in result["content"]
    assert "Employees get 25 vacation days." in result["content"]


def test_full_corpus_resource_name_is_passed_through(connection):
    node = VertexAIRagSearch(
        connection=connection,
        rag_corpus_id="projects/other-project/locations/europe-west3/ragCorpora/999",
        client=MagicMock(),
    )
    node.client.retrieve_contexts.return_value = _proto_response([])

    node.execute(VertexAIRagSearchInputSchema(query="q"), RunnableConfig(callbacks=[]))

    request = node.client.retrieve_contexts.call_args.kwargs["request"]
    assert (
        request.vertex_rag_store.rag_resources[0].rag_corpus
        == "projects/other-project/locations/europe-west3/ragCorpora/999"
    )
    # parent must match the corpus location, not the connection's project/location
    assert request.parent == "projects/other-project/locations/europe-west3"


def test_client_only_node_with_full_resource_name_executes():
    node = VertexAIRagSearch(rag_corpus_id=CORPUS_NAME, client=MagicMock())
    node.client.retrieve_contexts.return_value = _proto_response([{"text": "Doc", "score": 0.5}])

    result = node.execute(VertexAIRagSearchInputSchema(query="q"), RunnableConfig(callbacks=[]))

    request = node.client.retrieve_contexts.call_args.kwargs["request"]
    assert request.parent == "projects/test-project/locations/us-central1"
    assert result["documents"][0].content == "Doc"


def test_client_only_node_with_bare_corpus_id_raises_recoverable_error():
    node = VertexAIRagSearch(rag_corpus_id="123", client=MagicMock())

    with pytest.raises(ToolExecutionException) as exc_info:
        node.execute(VertexAIRagSearchInputSchema(query="q"), RunnableConfig(callbacks=[]))

    assert "full" in str(exc_info.value)
    node.client.retrieve_contexts.assert_not_called()


def test_input_overrides_node_defaults(tool):
    tool.metadata_filter = 'source_display_name = "handbook.pdf"'
    tool.client.retrieve_contexts.return_value = _proto_response([])

    tool.execute(
        VertexAIRagSearchInputSchema(query="q", top_k=3, metadata_filter='category = "hr"'),
        RunnableConfig(callbacks=[]),
    )

    request = tool.client.retrieve_contexts.call_args.kwargs["request"]
    assert request.query.rag_retrieval_config.top_k == 3
    assert request.query.rag_retrieval_config.filter.metadata_filter == 'category = "hr"'


def test_thresholds_and_file_ids_are_sent(connection):
    node = VertexAIRagSearch(
        connection=connection,
        rag_corpus_id="123",
        rag_file_ids=["file-1", "file-2"],
        vector_distance_threshold=0.5,
        client=MagicMock(),
    )
    node.client.retrieve_contexts.return_value = _proto_response([{"text": "Doc", "score": 0.2}])

    result = node.execute(VertexAIRagSearchInputSchema(query="q"), RunnableConfig(callbacks=[]))

    request = node.client.retrieve_contexts.call_args.kwargs["request"]
    assert request.query.rag_retrieval_config.filter.vector_distance_threshold == pytest.approx(0.5)
    assert list(request.vertex_rag_store.rag_resources[0].rag_file_ids) == ["file-1", "file-2"]
    assert result["documents"][0].metadata["score_type"] == "distance"


def test_similarity_threshold_is_sent(connection):
    node = VertexAIRagSearch(
        connection=connection, rag_corpus_id="123", vector_similarity_threshold=0.7, client=MagicMock()
    )
    node.client.retrieve_contexts.return_value = _proto_response([{"text": "Doc", "score": 0.8}])

    result = node.execute(VertexAIRagSearchInputSchema(query="q"), RunnableConfig(callbacks=[]))

    request = node.client.retrieve_contexts.call_args.kwargs["request"]
    assert request.query.rag_retrieval_config.filter.vector_similarity_threshold == pytest.approx(0.7)
    assert result["documents"][0].metadata["score_type"] == "similarity"


def test_node_level_metadata_filter_is_used_when_input_has_none(tool):
    tool.metadata_filter = 'category = "hr"'
    tool.client.retrieve_contexts.return_value = _proto_response([])

    tool.execute(VertexAIRagSearchInputSchema(query="q"), RunnableConfig(callbacks=[]))

    request = tool.client.retrieve_contexts.call_args.kwargs["request"]
    assert request.query.rag_retrieval_config.filter.metadata_filter == 'category = "hr"'


def test_thresholds_are_mutually_exclusive(connection):
    with pytest.raises(ValidationError):
        VertexAIRagSearch(
            connection=connection,
            rag_corpus_id="123",
            vector_distance_threshold=0.5,
            vector_similarity_threshold=0.7,
            client=MagicMock(),
        )


def test_rankers_are_mutually_exclusive(connection):
    with pytest.raises(ValidationError):
        VertexAIRagSearch(
            connection=connection,
            rag_corpus_id="123",
            rank_service_model="semantic-ranker-default@latest",
            llm_ranker_model="gemini-2.0-flash",
            client=MagicMock(),
        )


def test_rank_service_ranking_is_sent(connection):
    node = VertexAIRagSearch(
        connection=connection,
        rag_corpus_id="123",
        rank_service_model="semantic-ranker-default@latest",
        client=MagicMock(),
    )
    node.client.retrieve_contexts.return_value = _proto_response([{"text": "Doc", "score": 0.9}])

    result = node.execute(VertexAIRagSearchInputSchema(query="q"), RunnableConfig(callbacks=[]))

    request = node.client.retrieve_contexts.call_args.kwargs["request"]
    assert request.query.rag_retrieval_config.ranking.rank_service.model_name == "semantic-ranker-default@latest"
    assert result["documents"][0].metadata["score_type"] == "ranked"


def test_llm_ranker_ranking_is_sent(connection):
    node = VertexAIRagSearch(
        connection=connection, rag_corpus_id="123", llm_ranker_model="gemini-2.0-flash", client=MagicMock()
    )
    node.client.retrieve_contexts.return_value = _proto_response([])

    node.execute(VertexAIRagSearchInputSchema(query="q"), RunnableConfig(callbacks=[]))

    request = node.client.retrieve_contexts.call_args.kwargs["request"]
    assert request.query.rag_retrieval_config.ranking.llm_ranker.model_name == "gemini-2.0-flash"


def test_score_absent_stays_none_and_zero_score_is_kept(tool):
    tool.client.retrieve_contexts.return_value = _proto_response(
        [
            {"text": "Unscored context"},
            {"text": "Zero-score context", "score": 0.0},
        ]
    )

    result = tool.execute(VertexAIRagSearchInputSchema(query="q"), RunnableConfig(callbacks=[]))

    assert result["documents"][0].score is None
    assert result["documents"][1].score == 0.0


def test_document_ids_are_deterministic(tool):
    response_specs = [{"source_uri": "gs://bucket/a.pdf", "text": "Chunk text", "score": 0.5}]
    tool.client.retrieve_contexts.return_value = _proto_response(response_specs)
    first = tool.execute(VertexAIRagSearchInputSchema(query="q"), RunnableConfig(callbacks=[]))

    tool.client.retrieve_contexts.return_value = _proto_response(response_specs)
    second = tool.execute(VertexAIRagSearchInputSchema(query="q"), RunnableConfig(callbacks=[]))

    assert first["documents"][0].id == second["documents"][0].id


def test_metadata_allowlist_filters_formatted_content(tool):
    tool.metadata_fields = ["source_display_name"]
    tool.client.retrieve_contexts.return_value = _proto_response(
        [{"source_uri": "gs://bucket/a.pdf", "source_display_name": "a.pdf", "text": "Doc", "score": 0.5}]
    )

    result = tool.execute(VertexAIRagSearchInputSchema(query="q"), RunnableConfig(callbacks=[]))

    assert "source_display_name: a.pdf" in result["content"]
    assert "rag_corpus" not in result["content"]
    # the documents payload still carries the full metadata
    assert result["documents"][0].metadata["rag_corpus"] == CORPUS_NAME


def test_metadata_fields_none_emits_all_metadata(tool):
    tool.metadata_fields = None
    tool.client.retrieve_contexts.return_value = _proto_response(
        [{"source_uri": "gs://bucket/a.pdf", "source_display_name": "a.pdf", "text": "Doc", "score": 0.5}]
    )

    result = tool.execute(VertexAIRagSearchInputSchema(query="q"), RunnableConfig(callbacks=[]))

    assert "source_uri: gs://bucket/a.pdf" in result["content"]
    assert "source_display_name: a.pdf" in result["content"]
    assert f"rag_corpus: {CORPUS_NAME}" in result["content"]


def test_malformed_response_raises_recoverable_error(tool):
    tool.client.retrieve_contexts.return_value = object()  # no .contexts attribute

    with pytest.raises(ToolExecutionException) as exc_info:
        tool.execute(VertexAIRagSearchInputSchema(query="q"), RunnableConfig(callbacks=[]))

    assert "Failed to parse response" in str(exc_info.value)


def test_empty_query_raises_recoverable_error(tool):
    with pytest.raises(ToolExecutionException):
        tool.execute(VertexAIRagSearchInputSchema(query="   "), RunnableConfig(callbacks=[]))
    tool.client.retrieve_contexts.assert_not_called()


def test_empty_results_return_friendly_message(tool):
    tool.client.retrieve_contexts.return_value = _proto_response([])

    result = tool.execute(VertexAIRagSearchInputSchema(query="q"), RunnableConfig(callbacks=[]))

    assert result["documents"] == []
    assert result["content"] == "No results found for the given query."


def test_api_error_raises_recoverable_tool_exception(tool):
    from google.api_core import exceptions as google_exceptions

    tool.client.retrieve_contexts.side_effect = google_exceptions.PermissionDenied("caller lacks permission")

    with pytest.raises(ToolExecutionException) as exc_info:
        tool.execute(VertexAIRagSearchInputSchema(query="q"), RunnableConfig(callbacks=[]))

    assert "caller lacks permission" in str(exc_info.value)


@pytest.mark.asyncio
async def test_execute_async_builds_request_and_parses_documents(tool):
    async_client = MagicMock()
    async_client.retrieve_contexts = AsyncMock(
        return_value=_proto_response([{"source_uri": "gs://bucket/a.pdf", "text": "Async doc", "score": 0.9}])
    )

    with patch.object(VertexAIRagSearch, "get_async_client", AsyncMock(return_value=async_client)):
        result = await tool.execute_async(VertexAIRagSearchInputSchema(query="hi"), RunnableConfig(callbacks=[]))

    call_kwargs = async_client.retrieve_contexts.call_args.kwargs
    assert call_kwargs["timeout"] == 30
    request = call_kwargs["request"]
    assert request.query.text == "hi"
    assert result["documents"][0].content == "Async doc"


@pytest.mark.asyncio
async def test_get_async_client_is_cached_within_a_loop(tool):
    first = await tool.get_async_client()
    second = await tool.get_async_client()

    assert first is second
    assert len(tool._async_clients) == 1


def test_get_async_client_replaces_entries_from_dead_loops(tool):
    import asyncio

    first = asyncio.run(tool.get_async_client())
    second = asyncio.run(tool.get_async_client())

    # a client bound to a closed loop must never be reused, and dead entries must be pruned
    assert first is not second
    assert len(tool._async_clients) == 1


def test_yaml_roundtrip(tmp_path, connection):
    node = VertexAIRagSearch(
        id="vertex-rag-node",
        connection=connection,
        rag_corpus_id="123",
        top_k=7,
        vector_distance_threshold=0.5,
        metadata_filter='category = "hr"',
        metadata_fields=["source_uri"],
    )
    workflow = Workflow(id="rag-workflow", flow=Flow(id="rag-flow", nodes=[node]))

    yaml_path = tmp_path / "vertex_rag_workflow.yaml"
    workflow.to_yaml_file(yaml_path)

    loaded = Workflow.from_yaml_file(str(yaml_path), init_components=True)
    loaded_node = loaded.flow.nodes[0]
    assert isinstance(loaded_node, VertexAIRagSearch)
    assert loaded_node.client is not None  # init_components built the GAPIC client from the credentials

    roundtrip_path = tmp_path / "vertex_rag_workflow_roundtrip.yaml"
    loaded.to_yaml_file(roundtrip_path)
    roundtrip_node = Workflow.from_yaml_file(str(roundtrip_path), init_components=True).flow.nodes[0]

    assert roundtrip_node.rag_corpus_id == "123"
    assert roundtrip_node.top_k == 7
    assert roundtrip_node.vector_distance_threshold == 0.5
    assert roundtrip_node.metadata_filter == 'category = "hr"'
    assert roundtrip_node.metadata_fields == ["source_uri"]
    assert roundtrip_node.connection.id == "vertex-conn"
    assert roundtrip_node.connection.vertex_project_location == "us-central1"

    # The deserialized node still executes end-to-end with the config it was loaded with.
    roundtrip_node.client = MagicMock()
    roundtrip_node.client.retrieve_contexts.return_value = _proto_response(
        [{"source_uri": "gs://bucket/a.pdf", "text": "hit", "score": 0.3}]
    )

    result = roundtrip_node.execute(VertexAIRagSearchInputSchema(query="q"), RunnableConfig(callbacks=[]))

    request = roundtrip_node.client.retrieve_contexts.call_args.kwargs["request"]
    assert request.query.rag_retrieval_config.top_k == 7  # node-level top_k survived the roundtrip
    assert request.query.rag_retrieval_config.filter.vector_distance_threshold == pytest.approx(0.5)
    assert request.vertex_rag_store.rag_resources[0].rag_corpus == CORPUS_NAME
    assert [doc.content for doc in result["documents"]] == ["hit"]
