from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError
from pydantic import ValidationError

from dynamiq import Workflow
from dynamiq.connections import AWS
from dynamiq.flows import Flow
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.knowledgebases.bedrock import (
    BedrockGuardrailConfig,
    BedrockImplicitFilterAttribute,
    BedrockImplicitFilterConfig,
    BedrockKnowledgeBaseSearch,
    BedrockKnowledgeBaseSearchInputSchema,
    BedrockManagedSearchConfig,
    BedrockRerankingConfig,
)
from dynamiq.runnables import RunnableConfig
from dynamiq.types import Document


@pytest.fixture
def connection():
    return AWS(access_key_id="test-key", secret_access_key="test-secret", region="us-east-1")


@pytest.fixture
def tool(connection):
    node = BedrockKnowledgeBaseSearch(connection=connection, knowledge_base_id="KB12345678")
    node.client = MagicMock()  # bypass the real bedrock-agent-runtime client
    return node


def _text_result(text, score=0.9, uri="s3://bucket/doc.pdf", metadata=None):
    return {
        "content": {"type": "TEXT", "text": text},
        "location": {"type": "S3", "s3Location": {"uri": uri}},
        "score": score,
        "metadata": metadata if metadata is not None else {"title": "Doc title"},
    }


def test_type_resolves_to_module_path(tool):
    assert tool.type == "dynamiq.nodes.knowledgebases.BedrockKnowledgeBaseSearch"


def test_query_is_required():
    with pytest.raises(ValidationError):
        BedrockKnowledgeBaseSearchInputSchema()


def test_user_id_is_hidden_from_agent_tool_schema():
    from dynamiq.nodes.agents.components.schema_generator import generate_function_calling_schemas

    tool_name = "bedrock-knowledgebase-search"
    tool = MagicMock()
    tool.name = tool_name
    tool.description = "test"
    tool.input_schema = BedrockKnowledgeBaseSearchInputSchema
    tool.resolved_input_schema = BedrockKnowledgeBaseSearchInputSchema

    schema = next(
        s
        for s in generate_function_calling_schemas(
            tools=[tool], delegation_allowed=False, sanitize_tool_name=lambda name: name
        )
        if s["function"]["name"] == tool_name
    )
    properties = schema["function"]["parameters"]["properties"]

    assert "user_id" not in properties
    assert "query" in properties
    assert "filters" in properties
    assert "top_k" in properties


def test_execute_builds_request_and_parses_documents(tool):
    tool.client.retrieve.return_value = {
        "retrievalResults": [
            _text_result(
                "Paris is the capital of France.",
                score=0.87,
                metadata={"x-amz-bedrock-kb-chunk-id": "chunk-1", "title": "Geo"},
            ),
            _text_result("France is in Europe.", score=0.61, uri="s3://bucket/other.pdf", metadata={}),
        ]
    }

    input_data = BedrockKnowledgeBaseSearchInputSchema(query="capital of France")
    result = tool.execute(input_data, RunnableConfig(callbacks=[]))

    _, kwargs = tool.client.retrieve.call_args
    assert kwargs["knowledgeBaseId"] == "KB12345678"
    assert kwargs["retrievalQuery"] == {"text": "capital of France"}
    assert kwargs["retrievalConfiguration"] == {"vectorSearchConfiguration": {"numberOfResults": 5}}
    assert "guardrailConfiguration" not in kwargs
    assert "userContext" not in kwargs

    documents = result["documents"]
    assert all(isinstance(doc, Document) for doc in documents)
    assert documents[0].id == "chunk-1"
    assert documents[0].score == 0.87
    assert documents[0].metadata["source_uri"] == "s3://bucket/doc.pdf"
    assert documents[0].metadata["location_type"] == "S3"
    assert documents[0].metadata["content_type"] == "TEXT"
    assert "--- Retrieved Source 1 ---" in result["content"]
    assert "Score: 0.87" in result["content"]
    assert "title: Geo" in result["content"]
    assert "source_uri: s3://bucket/doc.pdf" in result["content"]
    assert "Paris is the capital of France." in result["content"]


def test_formatted_content_respects_metadata_allowlist(tool):
    tool.client.retrieve.return_value = {
        "retrievalResults": [
            _text_result("Doc", metadata={"title": "A", "internal_field": "hidden", "source_uri": "s3://x"})
        ]
    }

    result = tool.execute(BedrockKnowledgeBaseSearchInputSchema(query="q"), RunnableConfig(callbacks=[]))

    assert "title: A" in result["content"]
    assert "internal_field" not in result["content"]


def test_metadata_fields_none_emits_all_metadata(tool):
    tool.metadata_fields = None
    tool.client.retrieve.return_value = {
        "retrievalResults": [_text_result("Doc", metadata={"custom_attribute": "value"})]
    }

    result = tool.execute(BedrockKnowledgeBaseSearchInputSchema(query="q"), RunnableConfig(callbacks=[]))

    assert "custom_attribute: value" in result["content"]


def test_input_overrides_node_defaults(tool):
    tool.filters = {"equals": {"key": "department", "value": "hr"}}
    tool.client.retrieve.return_value = {"retrievalResults": []}

    input_filters = {"equals": {"key": "year", "value": 2024}}
    tool.execute(
        BedrockKnowledgeBaseSearchInputSchema(query="q", top_k=3, filters=input_filters),
        RunnableConfig(callbacks=[]),
    )

    _, kwargs = tool.client.retrieve.call_args
    assert kwargs["retrievalConfiguration"] == {
        "vectorSearchConfiguration": {"numberOfResults": 3, "filter": input_filters}
    }


def test_node_default_filters_and_user_id_are_used(tool):
    tool.filters = {"equals": {"key": "department", "value": "hr"}}
    tool.user_id = "user@example.com"
    tool.client.retrieve.return_value = {"retrievalResults": []}

    tool.execute(BedrockKnowledgeBaseSearchInputSchema(query="q"), RunnableConfig(callbacks=[]))

    _, kwargs = tool.client.retrieve.call_args
    assert kwargs["retrievalConfiguration"]["vectorSearchConfiguration"]["filter"] == tool.filters
    assert kwargs["userContext"] == {"userId": "user@example.com"}


def test_search_type_override(tool):
    tool.search_type = "HYBRID"
    tool.client.retrieve.return_value = {"retrievalResults": []}

    tool.execute(BedrockKnowledgeBaseSearchInputSchema(query="q"), RunnableConfig(callbacks=[]))

    _, kwargs = tool.client.retrieve.call_args
    assert kwargs["retrievalConfiguration"]["vectorSearchConfiguration"]["overrideSearchType"] == "HYBRID"


@pytest.mark.parametrize(
    "filters",
    [
        {"unknownOp": {"key": "a", "value": 1}},
        {"andAll": [{"equals": {"key": "a", "value": 1}}]},  # group operators need at least 2 items
        {"equals": {"key": "a"}},  # leaf operators need both key and value
        {"equals": {"key": "a", "value": 1}, "notEquals": {"key": "b", "value": 2}},  # one operator per object
    ],
)
def test_invalid_filters_raise_recoverable_error(tool, filters):
    with pytest.raises(ToolExecutionException) as exc_info:
        tool.execute(BedrockKnowledgeBaseSearchInputSchema(query="q", filters=filters), RunnableConfig(callbacks=[]))

    assert "filter" in str(exc_info.value).lower()
    tool.client.retrieve.assert_not_called()


def test_nested_group_filters_are_valid(tool):
    tool.client.retrieve.return_value = {"retrievalResults": []}
    filters = {
        "andAll": [
            {"equals": {"key": "department", "value": "hr"}},
            {"orAll": [{"greaterThan": {"key": "year", "value": 2023}}, {"in": {"key": "tag", "value": ["a", "b"]}}]},
        ]
    }

    tool.execute(BedrockKnowledgeBaseSearchInputSchema(query="q", filters=filters), RunnableConfig(callbacks=[]))

    _, kwargs = tool.client.retrieve.call_args
    assert kwargs["retrievalConfiguration"]["vectorSearchConfiguration"]["filter"] == filters


def test_empty_query_raises_recoverable_error(tool):
    with pytest.raises(ToolExecutionException):
        tool.execute(BedrockKnowledgeBaseSearchInputSchema(query="   "), RunnableConfig(callbacks=[]))
    tool.client.retrieve.assert_not_called()


def test_reranking_and_implicit_filter_configuration(connection):
    node = BedrockKnowledgeBaseSearch(
        connection=connection,
        knowledge_base_id="KB12345678",
        reranking=BedrockRerankingConfig(
            model_arn="arn:aws:bedrock:us-west-2::foundation-model/amazon.rerank-v1:0",
            number_of_reranked_results=3,
            metadata_selection_mode="SELECTIVE",
            metadata_fields_to_include=["title"],
        ),
        implicit_filter=BedrockImplicitFilterConfig(
            metadata_attributes=[
                BedrockImplicitFilterAttribute(key="year", type="NUMBER", description="Publication year")
            ],
            model_arn="arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-3-5-sonnet-20241022-v2:0",
        ),
    )
    node.client = MagicMock()
    node.client.retrieve.return_value = {"retrievalResults": []}

    node.execute(BedrockKnowledgeBaseSearchInputSchema(query="q"), RunnableConfig(callbacks=[]))

    _, kwargs = node.client.retrieve.call_args
    vector_config = kwargs["retrievalConfiguration"]["vectorSearchConfiguration"]
    assert vector_config["rerankingConfiguration"] == {
        "type": "BEDROCK_RERANKING_MODEL",
        "bedrockRerankingConfiguration": {
            "modelConfiguration": {"modelArn": "arn:aws:bedrock:us-west-2::foundation-model/amazon.rerank-v1:0"},
            "numberOfRerankedResults": 3,
            "metadataConfiguration": {
                "selectionMode": "SELECTIVE",
                "selectiveModeConfiguration": {"fieldsToInclude": [{"fieldName": "title"}]},
            },
        },
    }
    assert vector_config["implicitFilterConfiguration"] == {
        "metadataAttributes": [{"key": "year", "type": "NUMBER", "description": "Publication year"}],
        "modelArn": "arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-3-5-sonnet-20241022-v2:0",
    }


def test_managed_search_configuration(connection):
    node = BedrockKnowledgeBaseSearch(
        connection=connection,
        knowledge_base_id="arn:aws:bedrock:us-east-1:123456789012:knowledge-base/KB12345678",
        managed_search=BedrockManagedSearchConfig(reranking_model_type="MANAGED"),
        user_id="user@example.com",
    )
    node.client = MagicMock()
    node.client.retrieve.return_value = {"retrievalResults": []}

    node.execute(BedrockKnowledgeBaseSearchInputSchema(query="q"), RunnableConfig(callbacks=[]))

    _, kwargs = node.client.retrieve.call_args
    assert kwargs["retrievalConfiguration"] == {
        "managedSearchConfiguration": {"numberOfResults": 5, "rerankingModelType": "MANAGED"}
    }
    assert kwargs["userContext"] == {"userId": "user@example.com"}


def test_managed_search_custom_reranking_configuration(connection):
    node = BedrockKnowledgeBaseSearch(
        connection=connection,
        knowledge_base_id="KB12345678",
        managed_search=BedrockManagedSearchConfig(
            reranking_model_type="CUSTOM",
            reranking=BedrockRerankingConfig(
                model_arn="arn:aws:bedrock:us-west-2::foundation-model/cohere.rerank-v3-5:0",
                metadata_selection_mode="SELECTIVE",
                metadata_fields_to_exclude=["internal_field"],
            ),
        ),
    )
    node.client = MagicMock()
    node.client.retrieve.return_value = {"retrievalResults": []}

    node.execute(BedrockKnowledgeBaseSearchInputSchema(query="q"), RunnableConfig(callbacks=[]))

    _, kwargs = node.client.retrieve.call_args
    assert kwargs["retrievalConfiguration"] == {
        "managedSearchConfiguration": {
            "numberOfResults": 5,
            "rerankingModelType": "CUSTOM",
            "rerankingConfiguration": {
                "type": "BEDROCK_RERANKING_MODEL",
                "bedrockRerankingConfiguration": {
                    "modelConfiguration": {
                        "modelArn": "arn:aws:bedrock:us-west-2::foundation-model/cohere.rerank-v3-5:0"
                    },
                    "metadataConfiguration": {
                        "selectionMode": "SELECTIVE",
                        "selectiveModeConfiguration": {"fieldsToExclude": [{"fieldName": "internal_field"}]},
                    },
                },
            },
        }
    }


@pytest.mark.parametrize(
    "config_kwargs",
    [
        {"reranking": {"model_arn": "arn:model"}},  # custom config without reranking_model_type='CUSTOM'
        {"reranking_model_type": "CUSTOM"},  # CUSTOM without a reranking config
    ],
)
def test_managed_search_reranking_validation_errors(config_kwargs):
    with pytest.raises(ValidationError):
        BedrockManagedSearchConfig(**config_kwargs)


@pytest.mark.parametrize(
    "config_kwargs",
    [
        # include and exclude are mutually exclusive
        {
            "metadata_selection_mode": "SELECTIVE",
            "metadata_fields_to_include": ["a"],
            "metadata_fields_to_exclude": ["b"],
        },
        # include/exclude require SELECTIVE mode
        {"metadata_fields_to_include": ["a"]},
        {"metadata_selection_mode": "ALL", "metadata_fields_to_exclude": ["b"]},
    ],
)
def test_reranking_metadata_selection_validation_errors(config_kwargs):
    with pytest.raises(ValidationError):
        BedrockRerankingConfig(model_arn="arn:model", **config_kwargs)


def test_managed_search_rejects_vector_only_options(connection):
    with pytest.raises(ValidationError):
        BedrockKnowledgeBaseSearch(
            connection=connection,
            knowledge_base_id="KB12345678",
            managed_search=BedrockManagedSearchConfig(),
            search_type="HYBRID",
        )


def test_retrieval_config_escape_hatch_wins(tool):
    tool.retrieval_config = {"vectorSearchConfiguration": {"numberOfResults": 42}}
    tool.client.retrieve.return_value = {"retrievalResults": []}

    tool.execute(BedrockKnowledgeBaseSearchInputSchema(query="q", top_k=3), RunnableConfig(callbacks=[]))

    _, kwargs = tool.client.retrieve.call_args
    assert kwargs["retrievalConfiguration"] == {"vectorSearchConfiguration": {"numberOfResults": 42}}


def test_guardrail_configuration_and_intervention_note(connection):
    node = BedrockKnowledgeBaseSearch(
        connection=connection,
        knowledge_base_id="KB12345678",
        guardrail_config=BedrockGuardrailConfig(guardrail_id="guardrail-1", guardrail_version="1"),
    )
    node.client = MagicMock()
    node.client.retrieve.return_value = {
        "retrievalResults": [_text_result("Allowed chunk")],
        "guardrailAction": "INTERVENED",
    }

    result = node.execute(BedrockKnowledgeBaseSearchInputSchema(query="q"), RunnableConfig(callbacks=[]))

    _, kwargs = node.client.retrieve.call_args
    assert kwargs["guardrailConfiguration"] == {"guardrailId": "guardrail-1", "guardrailVersion": "1"}
    assert result["guardrail_action"] == "INTERVENED"
    assert "guardrail intervened" in result["content"]


def test_row_content_is_rendered_readably(tool):
    tool.client.retrieve.return_value = {
        "retrievalResults": [
            {
                "content": {
                    "type": "ROW",
                    "row": [
                        {"columnName": "city", "columnValue": "Paris", "type": "STRING"},
                        {"columnName": "population", "columnValue": "2100000", "type": "LONG"},
                    ],
                },
                "location": {"type": "SQL", "sqlLocation": {"query": "SELECT ..."}},
            }
        ]
    }

    result = tool.execute(BedrockKnowledgeBaseSearchInputSchema(query="q"), RunnableConfig(callbacks=[]))

    document = result["documents"][0]
    assert document.content == "city: Paris\npopulation: 2100000"
    assert document.score is None  # structured results carry no score and it must stay None
    assert document.metadata["source_uri"] == "SELECT ..."


def test_image_content_is_never_inlined(tool):
    tool.client.retrieve.return_value = {
        "retrievalResults": [
            {
                "content": {"type": "IMAGE", "byteContent": "aGVsbG8="},
                "location": {"type": "S3", "s3Location": {"uri": "s3://bucket/diagram.png"}},
                "score": 0.5,
            }
        ]
    }

    result = tool.execute(BedrockKnowledgeBaseSearchInputSchema(query="q"), RunnableConfig(callbacks=[]))

    assert result["documents"][0].content == "[image content from s3://bucket/diagram.png]"
    assert "aGVsbG8=" not in result["content"]


def test_audio_content_uses_transcription(tool):
    tool.client.retrieve.return_value = {
        "retrievalResults": [
            {
                "content": {"type": "AUDIO", "audio": {"s3Uri": "s3://bucket/a.mp3", "transcription": "Hello world"}},
                "location": {"type": "S3", "s3Location": {"uri": "s3://bucket/a.mp3"}},
            }
        ]
    }

    result = tool.execute(BedrockKnowledgeBaseSearchInputSchema(query="q"), RunnableConfig(callbacks=[]))

    assert result["documents"][0].content == "Hello world"


def test_unknown_location_type_degrades_gracefully(tool):
    tool.client.retrieve.return_value = {
        "retrievalResults": [
            {
                "content": {"type": "TEXT", "text": "Doc"},
                "location": {"type": "FUTURE_SOURCE", "futureSourceLocation": {"url": "https://example.com/doc"}},
            }
        ]
    }

    result = tool.execute(BedrockKnowledgeBaseSearchInputSchema(query="q"), RunnableConfig(callbacks=[]))

    assert result["documents"][0].metadata["source_uri"] == "https://example.com/doc"


def test_score_threshold_keeps_unscored_documents(tool):
    tool.score_threshold = 0.5
    tool.client.retrieve.return_value = {
        "retrievalResults": [
            _text_result("High", score=0.9),
            _text_result("Low", score=0.2),
            {
                "content": {"type": "TEXT", "text": "Unscored"},
                "location": {"type": "S3", "s3Location": {"uri": "s3://x"}},
            },
        ]
    }

    result = tool.execute(BedrockKnowledgeBaseSearchInputSchema(query="q"), RunnableConfig(callbacks=[]))

    assert [doc.content for doc in result["documents"]] == ["High", "Unscored"]


def test_empty_results_return_friendly_message(tool):
    tool.client.retrieve.return_value = {"retrievalResults": []}

    result = tool.execute(BedrockKnowledgeBaseSearchInputSchema(query="q"), RunnableConfig(callbacks=[]))

    assert result["documents"] == []
    assert result["content"] == "No results found for the given query."


def test_client_error_raises_recoverable_with_error_code(tool):
    tool.client.retrieve.side_effect = ClientError(
        error_response={"Error": {"Code": "ResourceNotFoundException", "Message": "KB not found"}},
        operation_name="Retrieve",
    )

    with pytest.raises(ToolExecutionException) as exc_info:
        tool.execute(BedrockKnowledgeBaseSearchInputSchema(query="q"), RunnableConfig(callbacks=[]))

    assert "ResourceNotFoundException" in str(exc_info.value)


def test_yaml_roundtrip(tmp_path):
    connection = AWS(id="aws-conn", access_key_id="test-key", secret_access_key="test-secret", region="us-east-1")
    node = BedrockKnowledgeBaseSearch(
        id="bedrock-kb-node",
        connection=connection,
        knowledge_base_id="KB12345678",
        top_k=7,
        search_type="SEMANTIC",
        score_threshold=0.4,
        metadata_fields=["title", "source_uri"],
        reranking=BedrockRerankingConfig(
            model_arn="arn:aws:bedrock:us-west-2::foundation-model/amazon.rerank-v1:0"
        ),
    )
    workflow = Workflow(id="kb-workflow", flow=Flow(id="kb-flow", nodes=[node]))

    yaml_path = tmp_path / "bedrock_kb_workflow.yaml"
    workflow.to_yaml_file(yaml_path)

    loaded = Workflow.from_yaml_file(str(yaml_path), init_components=True)
    loaded_node = loaded.flow.nodes[0]
    assert isinstance(loaded_node, BedrockKnowledgeBaseSearch)
    assert loaded_node.client is not None  # init_components built the bedrock-agent-runtime client

    roundtrip_path = tmp_path / "bedrock_kb_workflow_roundtrip.yaml"
    loaded.to_yaml_file(roundtrip_path)
    roundtrip_node = Workflow.from_yaml_file(str(roundtrip_path), init_components=True).flow.nodes[0]

    assert roundtrip_node.knowledge_base_id == "KB12345678"
    assert roundtrip_node.top_k == 7
    assert roundtrip_node.search_type == "SEMANTIC"
    assert roundtrip_node.score_threshold == 0.4
    assert roundtrip_node.metadata_fields == ["title", "source_uri"]
    assert roundtrip_node.reranking.model_arn == "arn:aws:bedrock:us-west-2::foundation-model/amazon.rerank-v1:0"
    assert roundtrip_node.connection.id == "aws-conn"
    assert roundtrip_node.connection.region == "us-east-1"

    # The deserialized node still executes end-to-end with the config it was loaded with.
    roundtrip_node.client = MagicMock()
    roundtrip_node.client.retrieve.return_value = {"retrievalResults": [_text_result("hit", score=0.8)]}

    result = roundtrip_node.execute(BedrockKnowledgeBaseSearchInputSchema(query="q"), RunnableConfig(callbacks=[]))

    _, kwargs = roundtrip_node.client.retrieve.call_args
    assert kwargs["knowledgeBaseId"] == "KB12345678"
    vector_config = kwargs["retrievalConfiguration"]["vectorSearchConfiguration"]
    assert vector_config["numberOfResults"] == 7  # node-level top_k survived the roundtrip
    assert vector_config["overrideSearchType"] == "SEMANTIC"
    assert [doc.content for doc in result["documents"]] == ["hit"]
