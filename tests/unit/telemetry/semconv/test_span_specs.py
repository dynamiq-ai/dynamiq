from typing import Any

import pytest
from opentelemetry.trace import SpanKind

import dynamiq.nodes.embedders as embedders
import dynamiq.nodes.llms as llms
from dynamiq import Workflow, connections
from dynamiq.flows import Flow
from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms.base import BaseLLM
from dynamiq.nodes.node import NodeDependency
from dynamiq.nodes.retrievers import (
    ChromaDocumentRetriever,
    PGVectorDocumentRetriever,
    PineconeDocumentRetriever,
    VectorStoreRetriever,
)
from dynamiq.nodes.tools.agent_tool import SubAgentTool
from dynamiq.nodes.utils import Input, Output
from dynamiq.telemetry import semconv
from dynamiq.telemetry.types import RunKind, SpanType

from .helpers import assert_otel_valid


class GroupNode(Node):
    """A node of any group, for mapping by group alone."""

    group: NodeGroup = NodeGroup.OPERATORS

    def execute(self, input_data: Any, config: Any = None, **kwargs) -> dict:
        return {}


def openai_llm(**kwargs) -> llms.OpenAI:
    return llms.OpenAI(model="gpt-4o", connection=connections.OpenAI(api_key="test-key"), **kwargs)


def spec_of(node, run_kwargs=None):
    spec = semconv.node_span_spec(node, run_id="run-1", run_kwargs=run_kwargs or {})
    assert_otel_valid(spec.attributes)
    return spec


@pytest.mark.parametrize(
    ("group", "span_type", "operation"),
    [
        (NodeGroup.LLMS, SpanType.CHAT, "chat"),
        (NodeGroup.EMBEDDERS, SpanType.EMBEDDINGS, "embeddings"),
        (NodeGroup.RETRIEVERS, SpanType.RETRIEVAL, "retrieval"),
        (NodeGroup.TOOLS, SpanType.TOOL, "execute_tool"),
        (NodeGroup.AGENTS, SpanType.AGENT, "invoke_agent"),
    ]
    + [
        (group, SpanType.NODE, None)
        for group in NodeGroup
        if group not in (NodeGroup.LLMS, NodeGroup.EMBEDDERS, NodeGroup.RETRIEVERS, NodeGroup.TOOLS, NodeGroup.AGENTS)
    ],
    ids=lambda value: value.value if isinstance(value, NodeGroup) else None,
)
def test_each_node_group_maps_to_its_span_type(group, span_type, operation):
    node = GroupNode(name="step", group=group)

    spec = spec_of(node)

    assert spec.span_type == span_type
    assert spec.attributes.get("gen_ai.operation.name") == operation
    assert spec.attributes["dynamiq.node.group"] == group.value
    assert spec.kind == (
        SpanKind.CLIENT if span_type in (SpanType.CHAT, SpanType.EMBEDDINGS, SpanType.RETRIEVAL) else SpanKind.INTERNAL
    )


def test_plain_node_span_has_node_identity_only():
    node = Input(name="start")

    spec = spec_of(node)

    assert spec.name == "start"
    assert spec.attributes == {
        "dynamiq.run.id": "run-1",
        "dynamiq.run.type": "node",
        "dynamiq.run.is_root": True,
        "dynamiq.node.id": node.id,
        "dynamiq.node.name": "start",
        "dynamiq.node.type": "dynamiq.nodes.utils.Input",
        "dynamiq.node.group": "utils",
    }


def test_chat_span_names_the_model_without_its_prefix():
    spec = spec_of(openai_llm())

    assert spec.name == "chat gpt-4o"
    assert spec.kind == SpanKind.CLIENT
    assert spec.attributes["gen_ai.operation.name"] == "chat"
    assert spec.attributes["gen_ai.provider.name"] == "openai"
    assert spec.attributes["gen_ai.system"] == "openai"
    assert spec.attributes["gen_ai.request.model"] == "gpt-4o"
    assert spec.attributes["dynamiq.node.model"] == "openai/gpt-4o"
    assert "dynamiq.llm.is_fallback" not in spec.attributes


def test_fallback_llm_run_is_flagged():
    llm = openai_llm()
    llm._is_fallback_run = True

    assert spec_of(llm).attributes["dynamiq.llm.is_fallback"] is True


def test_dependencies_of_the_definition_and_of_the_run():
    first, second = Input(), Output()
    node = Output(depends=[NodeDependency(first), NodeDependency(second)])
    run_depends = [NodeDependency(node=first).to_dict(for_tracing=True)]

    spec = spec_of(node, {"run_depends": run_depends})

    assert spec.attributes["dynamiq.node.depends"] == [first.id, second.id]
    assert spec.attributes["dynamiq.run.depends"] == [first.id]


EXPECTED_LLM_PROVIDERS = {
    "AI21": "ai21",
    "Anthropic": "anthropic",
    "Anyscale": "anyscale",
    "AtlasCloud": "atlascloud",
    "AzureAI": "azure.ai.openai",
    "Bedrock": "aws.bedrock",
    "Cerebras": "cerebras",
    "Cohere": "cohere",
    "CustomLLM": "openrouter",
    "Databricks": "databricks",
    "DeepInfra": "deepinfra",
    "DeepSeek": "deepseek",
    "FireworksAI": "fireworks_ai",
    "Gemini": "gcp.gemini",
    "Groq": "groq",
    "HuggingFace": "huggingface",
    "Mistral": "mistral_ai",
    "Novita": "novita",
    "NvidiaNIM": "nvidia_nim",
    "Ollama": "ollama",
    "OpenAI": "openai",
    "OpenRouter": "openrouter",
    "Perplexity": "perplexity",
    "Replicate": "replicate",
    "SambaNova": "sambanova",
    "TogetherAI": "together_ai",
    "VertexAI": "gcp.vertex_ai",
    "WatsonX": "ibm.watsonx.ai",
    "xAI": "x_ai",
}


def test_every_llm_class_has_an_expected_provider():
    classes = {
        name
        for name in llms.__all__
        if isinstance(getattr(llms, name), type) and issubclass(getattr(llms, name), BaseLLM) and name != "BaseLLM"
    }

    assert classes == set(EXPECTED_LLM_PROVIDERS)


@pytest.mark.parametrize(("class_name", "provider"), sorted(EXPECTED_LLM_PROVIDERS.items()))
def test_llm_provider_and_request_model(class_name, provider):
    cls = getattr(llms, class_name)
    model = f"{cls.MODEL_PREFIX or ''}some-model"
    extra = {"provider_prefix": "openrouter"} if class_name == "CustomLLM" else {}
    node = cls.model_construct(model=model, **extra)

    spec = spec_of(node)

    assert spec.attributes["gen_ai.provider.name"] == provider
    assert spec.attributes["gen_ai.system"] == provider
    assert spec.attributes["gen_ai.request.model"] == "some-model"
    assert spec.attributes["dynamiq.node.model"] == model
    assert spec.name == "chat some-model"


def test_llm_without_a_dynamiq_prefix_takes_the_provider_from_the_model():
    node = llms.CustomLLM.model_construct(model="together_ai/meta-llama/Llama-3")

    attributes = spec_of(node).attributes

    assert attributes["gen_ai.provider.name"] == "together_ai"
    assert attributes["gen_ai.request.model"] == "meta-llama/Llama-3"


def test_real_constructed_llm_nodes_strip_the_prefix_they_add():
    node = llms.Anthropic(model="claude-sonnet-4", connection=connections.Anthropic(api_key="test-key"))

    attributes = spec_of(node).attributes

    assert attributes["gen_ai.request.model"] == "claude-sonnet-4"
    assert attributes["dynamiq.node.model"] == "anthropic/claude-sonnet-4"


EXPECTED_EMBEDDER_PROVIDERS = {
    "Bedrock": ("aws.bedrock", "amazon.titan-embed-text-v1"),
    "Cohere": ("cohere", "embed-v4.0"),
    "Gemini": ("gcp.gemini", "gemini-embedding-001"),
    "HuggingFace": ("huggingface", None),
    "Mistral": ("mistral_ai", "mistral-embed"),
    "OpenAI": ("openai", "text-embedding-3-small"),
    "VertexAI": ("gcp.vertex_ai", "text-embedding-005"),
    "WatsonX": ("ibm.watsonx.ai", "ibm/slate-30m-english-rtrvr"),
}


@pytest.mark.parametrize("class_name", sorted(embedders.__all__))
def test_embedder_provider_and_request_model(class_name):
    cls = getattr(embedders, class_name)
    node = cls.model_construct()
    provider, request_model = EXPECTED_EMBEDDER_PROVIDERS[
        class_name.removesuffix("DocumentEmbedder").removesuffix("TextEmbedder")
    ]

    spec = spec_of(node)

    assert spec.span_type == SpanType.EMBEDDINGS
    assert spec.kind == SpanKind.CLIENT
    assert spec.attributes["gen_ai.operation.name"] == "embeddings"
    assert spec.attributes["gen_ai.provider.name"] == provider
    assert spec.attributes["gen_ai.system"] == provider
    # Default model strings carry the provider prefix for some embedders only; HuggingFace models keep their org.
    expected_model = request_model or node.model.removeprefix("huggingface/")
    assert spec.attributes["gen_ai.request.model"] == expected_model
    assert spec.name == f"embeddings {expected_model}"


@pytest.mark.parametrize(
    ("node", "data_source"),
    [
        (ChromaDocumentRetriever.model_construct(index_name="docs"), "docs"),
        (PineconeDocumentRetriever.model_construct(index_name="kb", namespace="ns"), "kb"),
        (PGVectorDocumentRetriever.model_construct(table_name="chunks", index_name="chunks_hnsw"), "chunks"),
    ],
    ids=["chroma", "pinecone", "pgvector"],
)
def test_retrieval_span_names_the_data_source(node, data_source):
    spec = spec_of(node)

    assert spec.name == f"retrieval {data_source}"
    assert spec.kind == SpanKind.CLIENT
    assert spec.attributes["gen_ai.data_source.id"] == data_source


def test_retrieval_span_without_a_data_source():
    node = GroupNode(group=NodeGroup.RETRIEVERS)

    spec = spec_of(node)

    assert spec.name == "retrieval"
    assert "gen_ai.data_source.id" not in spec.attributes


def test_vector_store_retriever_is_a_tool_span():
    node = VectorStoreRetriever.model_construct(name="kb-search")

    spec = spec_of(node)

    assert spec.span_type == SpanType.TOOL
    assert spec.name == "execute_tool kb-search"
    assert spec.attributes["gen_ai.tool.name"] == "kb-search"
    assert spec.attributes["gen_ai.tool.type"] == "function"


def test_agent_span_carries_its_llm():
    agent = Agent(name="support-agent", llm=openai_llm(), role="Helps.")

    spec = spec_of(agent)

    assert spec.name == "invoke_agent support-agent"
    assert spec.kind == SpanKind.INTERNAL
    assert spec.attributes["gen_ai.agent.name"] == "support-agent"
    assert spec.attributes["gen_ai.agent.id"] == agent.id
    assert spec.attributes["gen_ai.provider.name"] == "openai"
    assert spec.attributes["gen_ai.request.model"] == "gpt-4o"


def test_sub_agent_tool_is_an_agent_span():
    agent = Agent(name="researcher", llm=openai_llm())
    tool = SubAgentTool(name="Researcher", description="Researches.", agent=agent)

    spec = spec_of(tool)

    assert spec.name == "invoke_agent Researcher"
    assert spec.attributes["gen_ai.provider.name"] == "openai"


def test_workflow_span_spec():
    flow = Flow(nodes=[Input()])
    workflow = Workflow(name="support-flow", flow=flow, version="3")

    spec = semconv.workflow_span_spec(
        workflow, flow, run_id="wf-run", run_kind=RunKind.WORKFLOW, run_kwargs={"wf_run_id": "user-run"}
    )

    assert_otel_valid(spec.attributes)
    assert spec.span_type == SpanType.WORKFLOW
    assert spec.name == "invoke_workflow support-flow"
    assert spec.kind == SpanKind.INTERNAL
    assert spec.attributes == {
        "dynamiq.run.id": "wf-run",
        "dynamiq.run.type": "workflow",
        "dynamiq.run.is_root": True,
        "gen_ai.operation.name": "invoke_workflow",
        "gen_ai.workflow.name": "support-flow",
        "dynamiq.workflow.id": workflow.id,
        "dynamiq.workflow.version": "3",
        "dynamiq.flow.id": flow.id,
        "dynamiq.config.run_id": "user-run",
    }


def test_standalone_flow_span_spec():
    flow = Flow(name="sub", nodes=[Input()])

    spec = semconv.workflow_span_spec(None, flow, run_id="flow-run", run_kind=RunKind.FLOW, run_kwargs={})

    assert spec.name == "invoke_workflow sub"
    assert spec.attributes == {
        "dynamiq.run.id": "flow-run",
        "dynamiq.run.type": "flow",
        "dynamiq.run.is_root": True,
        "gen_ai.operation.name": "invoke_workflow",
        "gen_ai.workflow.name": "sub",
        "dynamiq.flow.id": flow.id,
    }


def test_span_specs_serialize_nothing(mocker):
    """The specs run before the sampling decision, so they must not serialize the node or its data."""
    to_dict = mocker.spy(llms.OpenAI, "to_dict")
    format_value = mocker.patch("dynamiq.telemetry.content.format_value")
    to_json = mocker.patch("dynamiq.telemetry.content.to_json")
    flow = Flow(nodes=[Input()])

    semconv.node_span_spec(Agent(llm=openai_llm()), run_id="r", run_kwargs={"run_depends": []})
    semconv.node_span_spec(openai_llm(), run_id="r", run_kwargs={})
    semconv.workflow_span_spec(Workflow(flow=flow), flow, run_id="r", run_kind=RunKind.WORKFLOW, run_kwargs={})

    to_dict.assert_not_called()
    format_value.assert_not_called()
    to_json.assert_not_called()


def test_scope_attributes():
    assert semconv.scope_attributes() == {"dynamiq.trace.shape_version": 1}
    assert semconv.SEMCONV_SCHEMA_URL == "https://opentelemetry.io/schemas/1.41.1"
