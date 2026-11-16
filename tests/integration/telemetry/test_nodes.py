"""Embedder, retriever, fallback, operators, mocked and cached runs, outcomes: one test per spec table row."""

from typing import Any
from unittest.mock import MagicMock

import pytest
from litellm import ModelResponse
from litellm.exceptions import RateLimitError
from litellm.types.utils import EmbeddingResponse
from opentelemetry.trace import SpanKind, StatusCode

from dynamiq import Workflow, connections
from dynamiq.components.retrievers.chroma import ChromaDocumentRetriever as ChromaRetrieverComponent
from dynamiq.flows import Flow
from dynamiq.nodes import ErrorHandling, Node, NodeGroup
from dynamiq.nodes.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.llms.base import FallbackConfig
from dynamiq.nodes.node import InputTransformer, NodeDependency
from dynamiq.nodes.operators.operators import Choice, ChoiceCondition, ChoiceOption, ConditionOperator, Map, Pass
from dynamiq.nodes.operators.sub_workflow import SubWorkflow
from dynamiq.nodes.retrievers import ChromaDocumentRetriever
from dynamiq.nodes.retrievers.retriever import VectorStoreRetriever
from dynamiq.nodes.utils import Input
from dynamiq.prompts import Message, MessageRole, Prompt
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.storages.vector import ChromaVectorStore
from dynamiq.types import Document
from dynamiq.types.mocking import MockConfig
from tests.integration.telemetry.conftest import OPERATION, RUN_ID, RUN_STATUS, attrs_json


class Lookup(Node):
    """A tool that records how often it really ran."""

    group: NodeGroup = NodeGroup.TOOLS
    name: str = "lookup"
    calls: int = 0

    def execute(self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs):
        self.calls += 1
        return {"content": "real result"}


class Flaky(Node):
    group: NodeGroup = NodeGroup.UTILS
    name: str = "flaky"
    fail_times: int = 1
    attempts: int = 0

    def execute(self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs):
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise ValueError(f"attempt {self.attempts} failed")
        return {"attempts": self.attempts}


class Failing(Node):
    group: NodeGroup = NodeGroup.UTILS
    name: str = "failing"

    def execute(self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs):
        raise ValueError("it broke")


def openai_llm(**kwargs) -> OpenAI:
    kwargs.setdefault("model", "gpt-4o")
    return OpenAI(
        connection=connections.OpenAI(api_key="test-key"),
        prompt=Prompt(messages=[Message(role=MessageRole.USER, content="Say hi")]),
        **kwargs,
    )


# Embedders


@pytest.fixture
def embedding_response(mocker):
    """An embedding call that reports the model and the usage the way the provider does."""

    def response(*args, **kwargs):
        embedded = EmbeddingResponse()
        count = len(kwargs.get("input") or [1])
        embedded["data"] = [{"embedding": [0.1, 0.2, 0.3], "index": i} for i in range(count)]
        embedded["model"] = "text-embedding-3-small-0125"
        embedded["usage"] = {"prompt_tokens": 6, "total_tokens": 6}
        return embedded

    return mocker.patch("dynamiq.components.embedders.base.BaseEmbedder._embedding", side_effect=response)


def test_text_embedder_span(mock_embedding_executor, config, trace):
    embedder = OpenAITextEmbedder(
        connection=connections.OpenAI(api_key="test-key"), model="text-embedding-3-small", name="emb"
    )

    result = Workflow(flow=Flow(nodes=[embedder])).run(input_data={"query": "find me"}, config=config)

    assert result.status == RunnableStatus.SUCCESS
    span = trace().one("embeddings text-embedding-3-small")
    assert span.kind == SpanKind.CLIENT
    a = span.attributes
    assert a[OPERATION] == "embeddings"
    assert a["gen_ai.provider.name"] == "openai"
    assert a["gen_ai.system"] == "openai"
    assert a["gen_ai.request.model"] == "text-embedding-3-small"
    assert a["gen_ai.embeddings.dimension.count"] == 1  # the length of the returned vector
    assert a["dynamiq.node.group"] == "embedders"
    assert a["dynamiq.node.model"].endswith("text-embedding-3-small")
    assert attrs_json(span, "dynamiq.input") == {"query": "find me"}


def test_document_embedder_reports_response_model_and_usage(embedding_response, config, trace):
    embedder = OpenAIDocumentEmbedder(
        connection=connections.OpenAI(api_key="test-key"), model="text-embedding-3-small", dimensions=3
    )

    result = Workflow(flow=Flow(nodes=[embedder])).run(
        input_data={"documents": [Document(content="one"), Document(content="two")]}, config=config
    )

    assert result.status == RunnableStatus.SUCCESS
    spans = trace()
    span = spans.one("embeddings text-embedding-3-small")
    a = span.attributes
    assert a["gen_ai.response.model"] == "text-embedding-3-small-0125"
    assert a["gen_ai.usage.input_tokens"] == 6
    assert a["gen_ai.embeddings.dimension.count"] == 3  # the node's dimensions
    assert not [k for k in spans.one("invoke_workflow Workflow").attributes if "usage" in k]


# Retrievers


@pytest.fixture
def retrieval_node(mock_embedding_executor):
    store = MagicMock(spec=ChromaVectorStore)
    store.index_name = "kb-docs"
    store.client = MagicMock()
    component = MagicMock(spec=ChromaRetrieverComponent)
    component.run.return_value = {
        "documents": [
            Document(id="d1", content="alpha passage", score=0.9),
            Document(id="d2", content="beta passage", score=0.5),
        ]
    }
    embedder = OpenAITextEmbedder(
        connection=connections.OpenAI(api_key="test-key"), model="text-embedding-3-small", name="query-embedder"
    )
    retriever = ChromaDocumentRetriever(
        name="chroma", vector_store=store, index_name="kb-docs", top_k=2, document_retriever=component
    )
    return embedder, retriever


@pytest.mark.parametrize(("input_top_k", "expected"), [(3, 3), (None, 2)], ids=["from-input", "from-node"])
def test_retriever_node_span(retrieval_node, config, trace, input_top_k, expected):
    _, retriever = retrieval_node
    data = {"embedding": [0.1, 0.2], "query": "find alpha"} | ({"top_k": input_top_k} if input_top_k else {})

    result = Workflow(flow=Flow(nodes=[retriever])).run(input_data=data, config=config)

    assert result.status == RunnableStatus.SUCCESS
    span = trace().one("retrieval kb-docs")
    assert span.kind == SpanKind.CLIENT
    a = span.attributes
    assert a[OPERATION] == "retrieval"
    assert a["gen_ai.data_source.id"] == "kb-docs"
    assert a["gen_ai.request.top_k"] == expected
    assert a["gen_ai.retrieval.query.text"] == "find alpha"
    # ids and scores only: the content of the documents is in dynamiq.output
    assert attrs_json(span, "gen_ai.retrieval.documents") == [{"id": "d1", "score": 0.9}, {"id": "d2", "score": 0.5}]
    assert "alpha passage" in a["dynamiq.output"]
    assert "alpha passage" not in a["gen_ai.retrieval.documents"]
    assert a["dynamiq.node.group"] == "retrievers"


def test_vector_store_retriever_tool_contains_embedding_and_retrieval(retrieval_node, config, trace):
    embedder, retriever = retrieval_node
    tool = VectorStoreRetriever(text_embedder=embedder, document_retriever=retriever, top_k=2)

    result = Workflow(flow=Flow(nodes=[tool])).run(input_data={"query": "find alpha"}, config=config)

    assert result.status == RunnableStatus.SUCCESS
    spans = trace()
    tool_span = spans.one("execute_tool vector-store-retriever")
    assert tool_span.kind == SpanKind.INTERNAL
    assert tool_span.attributes[OPERATION] == "execute_tool"
    assert tool_span.attributes["gen_ai.tool.type"] == "function"
    assert attrs_json(tool_span, "gen_ai.tool.call.arguments")["query"] == "find alpha"
    assert "alpha passage" in attrs_json(tool_span, "gen_ai.tool.call.result")["content"]
    assert [s.name for s in spans.children(tool_span)] == ["embeddings text-embedding-3-small", "retrieval kb-docs"]
    assert spans.parent_of(tool_span).name == "invoke_workflow Workflow"
    # the retrieval follows the embedding it was given
    embedding, retrieval = spans.children(tool_span)
    assert list(retrieval.attributes["dynamiq.run.depends"]) == [embedding.attributes["dynamiq.node.id"]]


# LLM fallback


def test_llm_fallback_is_a_sibling_of_the_failed_primary(config, trace):
    primary = openai_llm(model="gpt-4o-mini", name="primary")
    fallback = openai_llm(model="gpt-4o", name="backup")
    primary.fallback = FallbackConfig(llm=fallback, enabled=True)
    primary._completion = MagicMock(
        side_effect=RateLimitError(message="slow down", model="gpt-4o-mini", llm_provider="openai")
    )
    reply = ModelResponse()
    reply["choices"][0]["message"]["content"] = "from the fallback"
    fallback._completion = MagicMock(return_value=reply)

    result = Workflow(flow=Flow(nodes=[primary])).run(input_data={}, config=config)

    assert result.status == RunnableStatus.SUCCESS
    spans = trace()
    root = spans.one("invoke_workflow Workflow")
    failed = spans.one("chat gpt-4o-mini")
    used = spans.one("chat gpt-4o")
    assert spans.parent_of(failed) is root
    assert spans.parent_of(used) is root
    assert failed.attributes[RUN_STATUS] == "failed"
    assert failed.attributes["error.type"] == "RateLimitError"
    assert failed.status.status_code == StatusCode.ERROR
    assert [e.name for e in failed.events if e.name == "exception"] == ["exception"]
    assert "dynamiq.llm.is_fallback" not in failed.attributes
    assert used.attributes["dynamiq.llm.is_fallback"] is True
    assert used.attributes[RUN_STATUS] == "succeeded"
    assert attrs_json(used, "gen_ai.output.messages")[0]["parts"] == [{"type": "text", "content": "from the fallback"}]
    assert failed.attributes[RUN_ID] != used.attributes[RUN_ID]
    # the workflow succeeded: it reports its own outcome, not the failed attempt's
    assert root.attributes[RUN_STATUS] == "succeeded"
    assert root.status.status_code == StatusCode.UNSET


# Operators


def test_map_has_one_child_per_item(config, trace):
    mapper = Map(id="map", name="mapper", node=Pass(name="item"), max_workers=4)
    items = [{"n": i} for i in range(12)]

    result = Workflow(flow=Flow(nodes=[mapper])).run(input_data={"input": items}, config=config)

    assert result.status == RunnableStatus.SUCCESS
    spans = trace()
    map_span = spans.one("mapper")
    assert OPERATION not in map_span.attributes
    children = spans.children(map_span)
    assert len(children) == 12
    assert {c.name for c in children} == {"item"}
    assert sorted(attrs_json(c, "dynamiq.input")["n"] for c in children) == list(range(12))
    assert len({c.attributes[RUN_ID] for c in children}) == 12
    assert spans.parent_of(map_span).name == "invoke_workflow Workflow"
    assert len(spans.spans) == 1 + 1 + 12


def test_map_of_llm_runs_one_chat_span_per_item(mock_llm_executor, config, trace):
    mapper = Map(id="map", name="mapper", node=openai_llm(), max_workers=3)

    Workflow(flow=Flow(nodes=[mapper])).run(input_data={"input": [{"a": 1}, {"a": 2}, {"a": 3}]}, config=config)

    spans = trace()
    chats = spans.named("chat gpt-4o")
    assert len(chats) == 3
    assert all(spans.parent_of(c) is spans.one("mapper") for c in chats)


def test_choice_taken_and_skipped_branches_both_have_spans(config, trace):
    start = Input(id="input", name="input")
    choice = Choice(
        id="choice",
        name="choice",
        options=[
            ChoiceOption(
                id="yes",
                name="yes",
                condition=ChoiceCondition(variable="$.flag", operator=ConditionOperator.BOOLEAN_EQUALS, value=True),
            ),
            ChoiceOption(id="no", name="no", condition=None),
        ],
        input_transformer=InputTransformer(selector={"flag": "$.input.output.flag"}),
        depends=[NodeDependency(start)],
    )
    taken = Pass(id="a", name="taken", depends=[NodeDependency(choice, option="yes")])
    skipped = Pass(id="b", name="skipped", depends=[NodeDependency(choice, option="no")])

    result = Workflow(flow=Flow(nodes=[start, choice, taken, skipped])).run(input_data={"flag": True}, config=config)

    assert result.status == RunnableStatus.SUCCESS
    spans = trace()
    root = spans.one("invoke_workflow Workflow")
    assert sorted(s.name for s in spans.children(root)) == ["choice", "input", "skipped", "taken"]
    assert spans.one("taken").attributes[RUN_STATUS] == "succeeded"
    skipped_span = spans.one("skipped")
    assert skipped_span.attributes[RUN_STATUS] == "skipped"
    assert skipped_span.status.status_code == StatusCode.UNSET
    skip = attrs_json(skipped_span, "dynamiq.skip")
    assert skip["failed_dependency"]["node"]["id"] == "choice"
    assert skip["failed_dependency"]["option"] == "no"
    assert list(skipped_span.attributes["dynamiq.node.depends"]) == ["choice"]
    assert "dynamiq.output" not in skipped_span.attributes
    # the Choice and its branches are plain spans: no GenAI operation
    assert OPERATION not in spans.one("choice").attributes


def test_sub_workflow_node_nests_a_flow_span(config, trace):
    inner = Flow(id="inner-flow", nodes=[Input(id="inner-in", name="inner-in")])
    node = SubWorkflow(id="sub", name="sub", flow=inner)

    result = Workflow(flow=Flow(nodes=[node])).run(input_data={"x": 1}, config=config)

    assert result.status == RunnableStatus.SUCCESS
    spans = trace()
    assert spans.shape() == [
        "invoke_workflow Workflow",
        [["sub", [["invoke_workflow Flow", [["inner-in", []]]]]]],
    ]
    flow_span = spans.one("invoke_workflow Flow")
    assert flow_span.attributes["dynamiq.run.type"] == "flow"
    assert flow_span.attributes["gen_ai.workflow.name"] == "Flow"
    assert flow_span.attributes["dynamiq.flow.id"] == "inner-flow"
    assert flow_span.attributes[OPERATION] == "invoke_workflow"
    assert "dynamiq.workflow.id" not in flow_span.attributes
    assert "dynamiq.flow.run_id" not in flow_span.attributes
    assert spans.one("invoke_workflow Workflow").attributes["dynamiq.run.type"] == "workflow"
    assert len({s.context.trace_id for s in spans.spans}) == 1


def test_standalone_flow_is_a_root_span(config, trace):
    flow = Flow(id="solo", nodes=[Input(id="i", name="in")])

    flow.run_sync(input_data={"v": 1}, config=config)

    root = trace().one("invoke_workflow Flow")
    assert root.parent is None
    assert root.attributes["dynamiq.run.type"] == "flow"
    assert root.attributes["dynamiq.flow.id"] == "solo"
    assert [s.name for s in trace().children(root)] == ["in"]


# Mocked and cached runs


def test_mocked_run_is_flagged_and_has_no_attempt(config, trace):
    tool = Lookup(mock=MockConfig(enabled=True, output={"content": "canned"}))

    result = Workflow(flow=Flow(nodes=[tool])).run(input_data={"q": "x"}, config=config)

    assert result.status == RunnableStatus.SUCCESS
    assert tool.calls == 0
    span = trace().one("execute_tool lookup")
    assert span.attributes["dynamiq.node.is_mocked"] is True
    assert "dynamiq.node.is_output_from_cache" not in span.attributes
    assert span.attributes[RUN_STATUS] == "succeeded"
    assert attrs_json(span, "gen_ai.tool.call.result")["content"] == "canned"
    assert not [e for e in span.events if e.name == "dynamiq.node.execution"]


def test_real_run_has_no_mock_or_cache_flag(config, trace):
    tool = Lookup()

    Workflow(flow=Flow(nodes=[tool])).run(input_data={"q": "x"}, config=config)

    attributes = trace().one("execute_tool lookup").attributes
    assert tool.calls == 1
    assert "dynamiq.node.is_mocked" not in attributes
    assert "dynamiq.node.is_output_from_cache" not in attributes


def test_cache_hit_is_flagged(mock_llm_executor, make_config, trace):
    from dynamiq.cache import RedisCacheConfig
    from dynamiq.nodes import CachingConfig

    llm = openai_llm(caching=CachingConfig(enabled=True), is_postponed_component_init=True)
    config = make_config()
    config.cache = RedisCacheConfig(host="redis-test-sv", port=6379, db=0, namespace="otel-test")

    first = llm.run(input_data={"a": 1}, config=config)
    second = llm.run(input_data={"a": 1}, config=config)

    assert first.status == second.status == RunnableStatus.SUCCESS
    assert mock_llm_executor.call_count == 1
    spans = sorted(trace().named("chat gpt-4o"), key=lambda s: s.start_time)
    assert len(spans) == 2
    miss, hit = spans
    assert "dynamiq.node.is_output_from_cache" not in miss.attributes
    assert hit.attributes["dynamiq.node.is_output_from_cache"] is True
    assert hit.attributes[RUN_STATUS] == "succeeded"
    assert not [e for e in hit.events if e.name == "dynamiq.node.execution"]
    assert attrs_json(hit, "dynamiq.output") == {"content": "mocked_response"}


# Outcomes


def test_retried_node_is_one_span_with_an_event_per_attempt(config, trace):
    node = Flaky(error_handling=ErrorHandling(max_retries=2, retry_interval_seconds=0.01))

    result = Workflow(flow=Flow(nodes=[node])).run(input_data={}, config=config)

    assert result.status == RunnableStatus.SUCCESS
    spans = trace()
    assert len(spans.named("flaky")) == 1
    span = spans.one("flaky")
    assert span.attributes[RUN_STATUS] == "succeeded"
    assert span.status.status_code == StatusCode.UNSET
    events = [e for e in span.events if e.name == "dynamiq.node.execution"]
    assert [e.attributes["dynamiq.execution.attempt"] for e in events] == [1, 2]
    assert [e.attributes["dynamiq.execution.status"] for e in events] == ["failed", "succeeded"]
    assert events[0].attributes["error.type"] == "ValueError"
    assert "attempt 1 failed" in events[0].attributes["exception.message"]
    assert "error.type" not in events[1].attributes
    assert all(e.attributes["dynamiq.execution.start_time_unix_nano"] > 0 for e in events)
    assert events[0].timestamp <= events[1].timestamp


def test_failed_node_fails_the_workflow(config, trace):
    start = Input(id="in", name="start")
    broken = Failing(id="broken", depends=[NodeDependency(start)])
    after = Pass(id="after", name="after", depends=[NodeDependency(broken)])

    result = Workflow(flow=Flow(nodes=[start, broken, after])).run(input_data={"v": 1}, config=config)

    assert result.status == RunnableStatus.FAILURE
    spans = trace()
    failed = spans.one("failing")
    assert failed.attributes[RUN_STATUS] == "failed"
    assert failed.attributes["error.type"] == "ValueError"
    assert failed.status.status_code == StatusCode.ERROR
    assert failed.status.description == "it broke"
    (event,) = (e for e in failed.events if e.name == "exception")
    assert event.attributes["exception.type"] == "ValueError"
    assert event.attributes["exception.message"] == "it broke"
    assert "Traceback" in event.attributes["exception.stacktrace"]
    skipped = spans.one("after")
    assert skipped.attributes[RUN_STATUS] == "skipped"
    assert attrs_json(skipped, "dynamiq.skip")["failed_dependency"]["node"]["id"] == "broken"
    root = spans.one("invoke_workflow Workflow")
    assert root.attributes[RUN_STATUS] == "failed"
    assert root.status.status_code == StatusCode.ERROR
    assert [f["id"] for f in attrs_json(root, "dynamiq.failed_nodes")] == ["broken"]
    assert attrs_json(root, "dynamiq.failed_nodes")[0]["error_message"] == "it broke"
