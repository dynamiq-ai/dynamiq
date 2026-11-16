import json
from typing import Any

import pytest

from dynamiq import Workflow, connections
from dynamiq.checkpoints.types import RunPausedException
from dynamiq.flows import Flow
from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from dynamiq.nodes.embedders.base import TextEmbeddingOutput
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.node import NodeDependency
from dynamiq.nodes.operators.operators import Choice, ChoiceCondition, ChoiceOption, ConditionOperator
from dynamiq.nodes.retrievers import ChromaDocumentRetriever
from dynamiq.nodes.utils import Input, Output
from dynamiq.prompts import Message, Prompt
from dynamiq.runnables import RunnableFailedNodeInfo
from dynamiq.telemetry import semconv
from dynamiq.telemetry.types import CaptureSettings, RunOutcome, SpanType
from dynamiq.types import Document
from dynamiq.types.llm_tool import Tool

from .helpers import assert_otel_valid, validate_messages

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the web.",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
    },
}
USAGE = {
    "prompt_tokens": 120,
    "prompt_tokens_cost_usd": 0.0003,
    "completion_tokens": 30,
    "completion_tokens_cost_usd": 0.0006,
    "total_tokens": 150,
    "total_tokens_cost_usd": 0.0009,
    "cache_read_input_tokens": 0,
    "cache_creation_input_tokens": None,
}


class ToolNode(Node):
    group: NodeGroup = NodeGroup.TOOLS
    name: str = "web-search"

    def execute(self, input_data: Any, config: Any = None, **kwargs) -> dict:
        return {}


def openai_llm(**kwargs) -> OpenAI:
    return OpenAI(model="gpt-4o", connection=connections.OpenAI(api_key="test-key"), **kwargs)


def checked(attributes: dict) -> dict:
    assert_otel_valid(attributes)
    return attributes


def call_input(span_type, entity, input_data, settings, run_kwargs=None, state=None):
    return checked(
        semconv.input_attributes(
            span_type,
            entity,
            input_data,
            run_kwargs=run_kwargs or {},
            settings=settings,
            state={} if state is None else state,
        )
    )


def call_execute_run(span_type, entity, run_kwargs, settings, state=None):
    return checked(
        semconv.execute_run_attributes(
            span_type, entity, run_kwargs=run_kwargs, settings=settings, state={} if state is None else state
        )
    )


def call_execute_output(span_type, entity, output, settings, state=None):
    return checked(
        semconv.execute_output_attributes(
            span_type, entity, output, settings=settings, state={} if state is None else state
        )
    )


def call_output(span_type, entity, output_data, settings, run_kwargs=None, state=None):
    return checked(
        semconv.output_attributes(
            span_type,
            entity,
            output_data,
            run_kwargs=run_kwargs or {},
            settings=settings,
            state={} if state is None else state,
        )
    )


# Every Dynamiq span


def test_run_attributes_repeat_the_run_settings():
    settings = CaptureSettings(
        session_id="s-1",
        user_id="u-1",
        source_id="src-1",
        tags=("prod", "eu"),
        metadata={"team": "core", "priority": 2, "ratio": 0.5, "beta": True},
    )

    assert checked(semconv.run_attributes(settings)) == {
        "gen_ai.conversation.id": "s-1",
        "session.id": "s-1",
        "user.id": "u-1",
        "dynamiq.source.id": "src-1",
        "dynamiq.tags": ["prod", "eu"],
        "dynamiq.metadata.team": "core",
        "dynamiq.metadata.priority": 2,
        "dynamiq.metadata.ratio": 0.5,
        "dynamiq.metadata.beta": True,
    }


def test_run_attributes_are_empty_without_settings(settings):
    assert semconv.run_attributes(settings) == {}


def test_agent_declares_its_conversation():
    agent = Agent(llm=openai_llm())

    assert semconv.conversation_of(SpanType.AGENT, agent, {"input": "q", "session_id": "s", "user_id": "u"}) == (
        "s",
        "u",
    )
    assert semconv.conversation_of(SpanType.AGENT, agent, {"input": "q"}) == (None, None)
    assert semconv.conversation_of(SpanType.TOOL, agent, {"session_id": "s"}) == (None, None)


@pytest.mark.parametrize(
    ("outcome", "run_kwargs", "expected"),
    [
        (RunOutcome.SUCCEEDED, {}, {"dynamiq.run.status": "succeeded"}),
        (
            RunOutcome.SUCCEEDED,
            {"is_mocked": True, "is_output_from_cache": False},
            {"dynamiq.run.status": "succeeded", "dynamiq.node.is_mocked": True},
        ),
        (
            RunOutcome.SUCCEEDED,
            {"is_output_from_cache": True},
            {"dynamiq.run.status": "succeeded", "dynamiq.node.is_output_from_cache": True},
        ),
        (RunOutcome.FAILED, {}, {"dynamiq.run.status": "failed"}),
        (RunOutcome.SKIPPED, {}, {"dynamiq.run.status": "skipped"}),
        (RunOutcome.CANCELED, {}, {"dynamiq.run.status": "canceled"}),
        (RunOutcome.PAUSED, {}, {"dynamiq.run.status": "paused"}),
    ],
)
def test_outcome_attributes(outcome, run_kwargs, expected):
    assert checked(semconv.outcome_attributes(outcome, run_kwargs=run_kwargs)) == expected


# Node spans


def test_node_input_and_output(settings):
    node = Output()

    assert call_input(SpanType.NODE, node, {"value": 1}, settings) == {"dynamiq.input": '{"value":1}'}
    assert call_output(SpanType.NODE, node, {"result": "ok"}, settings) == {"dynamiq.output": '{"result":"ok"}'}


def test_tool_input_and_output_use_the_tool_call_attributes(settings):
    node = ToolNode()

    assert call_input(SpanType.TOOL, node, {"q": "weather"}, settings) == {
        "gen_ai.tool.call.arguments": '{"q":"weather"}'
    }
    assert call_output(SpanType.TOOL, node, {"content": "sunny"}, settings) == {
        "gen_ai.tool.call.result": '{"content":"sunny"}'
    }


def test_node_config_is_recorded_without_prompt_messages():
    settings = CaptureSettings(node_config_enabled=True)
    llm = openai_llm(prompt=Prompt(messages=[Message(content="secret system prompt")]))

    attributes = call_input(SpanType.CHAT, llm, {}, settings)

    config = json.loads(attributes["dynamiq.node.config"])
    assert config["id"] == llm.id
    assert config["model"] == "openai/gpt-4o"
    assert "messages" not in config["prompt"]
    assert "secret system prompt" not in attributes["dynamiq.node.config"]
    assert "test-key" not in attributes["dynamiq.node.config"]


def test_node_config_is_off_by_default(settings):
    assert "dynamiq.node.config" not in call_input(SpanType.NODE, Input(), {}, settings)


def test_skip_attributes(settings):
    failed = Input(name="upstream")
    skip_data = {"failed_dependency": NodeDependency(node=failed).to_dict(for_tracing=True)}

    attributes = checked(
        semconv.skip_attributes(
            skip_data,
            {"q": "x"},
            "Not approved.",
            span_type=SpanType.NODE,
            entity=Output(),
            settings=settings,
            state={},
        )
    )

    assert json.loads(attributes["dynamiq.skip"])["failed_dependency"]["node"] == {
        "id": failed.id,
        "name": "upstream",
        "type": "dynamiq.nodes.utils.Input",
    }
    assert attributes["dynamiq.human_feedback"] == "Not approved."
    assert attributes["dynamiq.input"] == '{"q":"x"}'


def test_skip_without_content_keeps_only_the_dependency_identifiers(no_content):
    choice = Choice(id="choice", name="router", options=[ChoiceOption(id="no", name="no")])
    condition = ChoiceCondition(variable="$.output.x", operator=ConditionOperator.STRING_EQUALS, value="user-secret")
    skip_data = {
        "failed_dependency": NodeDependency(node=choice, option="no", condition=condition).to_dict(for_tracing=True)
    }
    kwargs = dict(span_type=SpanType.NODE, entity=Output(), state={})

    attributes = checked(semconv.skip_attributes(skip_data, {"q": "x"}, None, settings=no_content, **kwargs))

    assert json.loads(attributes["dynamiq.skip"]) == {
        "failed_dependency": {
            "node": {"id": "choice", "name": "router", "type": "dynamiq.nodes.operators.Choice"},
            "option": "no",
        }
    }
    with_content = checked(semconv.skip_attributes(skip_data, None, None, settings=CaptureSettings(), **kwargs))
    assert json.loads(with_content["dynamiq.skip"])["failed_dependency"]["condition"]["value"] == "user-secret"


def test_skipped_run_records_node_config_when_config_capture_is_on():
    skip_data = {"failed_dependency": NodeDependency(node=Input(name="upstream")).to_dict(for_tracing=True)}
    kwargs = dict(span_type=SpanType.NODE, entity=Output(name="sink"), state={})

    with_config = checked(
        semconv.skip_attributes(skip_data, None, None, settings=CaptureSettings(node_config_enabled=True), **kwargs)
    )
    without_config = semconv.skip_attributes(skip_data, None, None, settings=CaptureSettings(), **kwargs)

    # A skipped node never starts, so the skip is the only place its configuration can be recorded.
    assert json.loads(with_config["dynamiq.node.config"])["name"] == "sink"
    assert "dynamiq.node.config" not in without_config


def test_tool_data_is_recorded(settings):
    tool_data = {"tool_session_id": "sbx-1", "tool_session_host": "sbx-1.e2b.dev"}

    attributes = call_execute_run(SpanType.TOOL, ToolNode(), {"tool_data": tool_data}, settings)

    assert json.loads(attributes["dynamiq.tool.data"]) == tool_data


# Inference spans


def test_chat_request_parameters_that_are_set(settings):
    llm = openai_llm(
        temperature=0.2, max_tokens=256, top_p=0.9, frequency_penalty=0.1, presence_penalty=0, stop=["END"], seed=7
    )
    llm.streaming.enabled = True

    attributes = call_input(SpanType.CHAT, llm, {}, settings)

    assert attributes["gen_ai.request.temperature"] == 0.2
    assert attributes["gen_ai.request.max_tokens"] == 256
    assert attributes["gen_ai.request.top_p"] == 0.9
    assert attributes["gen_ai.request.frequency_penalty"] == 0.1
    assert attributes["gen_ai.request.presence_penalty"] == 0.0
    assert isinstance(attributes["gen_ai.request.presence_penalty"], float)
    assert attributes["gen_ai.request.stop_sequences"] == ["END"]
    assert attributes["gen_ai.request.seed"] == 7
    assert attributes["gen_ai.request.stream"] is True
    assert "gen_ai.output.type" not in attributes


def test_chat_request_parameters_that_are_not_set_are_left_out(settings):
    attributes = call_input(SpanType.CHAT, openai_llm(), {"x": 1}, settings)

    assert attributes == {"dynamiq.input": '{"x":1}'}


@pytest.mark.parametrize(
    "where",
    ["run_kwargs", "node", "prompt"],
)
def test_requested_response_format_sets_json_output_type(settings, where):
    response_format = {"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object"}}}
    llm = openai_llm(response_format=response_format if where == "node" else None)
    run_kwargs = {}
    if where == "run_kwargs":
        run_kwargs["response_format"] = response_format
    if where == "prompt":
        run_kwargs["prompt"] = Prompt(messages=[], response_format=response_format)

    assert call_input(SpanType.CHAT, llm, {}, settings, run_kwargs)["gen_ai.output.type"] == "json"


@pytest.mark.parametrize("where", ["run_kwargs", "node", "prompt"])
def test_tool_definitions_come_from_the_request(settings, where):
    llm = openai_llm(tools=[SEARCH_TOOL] if where == "node" else None)
    run_kwargs = {}
    if where == "run_kwargs":
        run_kwargs["tools"] = [SEARCH_TOOL]
    if where == "prompt":
        run_kwargs["prompt"] = Prompt(messages=[], tools=[Tool(**SEARCH_TOOL)])

    attributes = call_input(SpanType.CHAT, llm, {}, settings, run_kwargs)

    assert json.loads(attributes["gen_ai.tool.definitions"]) == [
        {
            "type": "function",
            "name": "search",
            "description": "Search the web.",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
        }
    ]


def test_prompt_messages_become_input_messages(settings):
    messages = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]

    attributes = call_execute_run(SpanType.CHAT, openai_llm(), {"prompt_messages": messages}, settings)

    assert validate_messages(attributes["gen_ai.input.messages"], "input") == [
        {"role": "system", "parts": [{"type": "text", "content": "Be brief."}]},
        {"role": "user", "parts": [{"type": "text", "content": "Hi"}]},
    ]


def test_response_data_and_usage(settings):
    state = {}
    run_kwargs = {
        "usage_data": USAGE,
        "response_data": {"id": "chatcmpl-1", "model": "gpt-4o-2024-08-06", "finish_reasons": ["tool_calls"]},
    }

    attributes = call_execute_run(SpanType.CHAT, openai_llm(), run_kwargs, settings, state)

    assert attributes == {
        "gen_ai.response.id": "chatcmpl-1",
        "gen_ai.response.model": "gpt-4o-2024-08-06",
        "gen_ai.response.finish_reasons": ["tool_call"],
        "gen_ai.usage.input_tokens": 120,
        "gen_ai.usage.output_tokens": 30,
        "dynamiq.usage.total_tokens": 150,
        "dynamiq.usage.input_cost_usd": 0.0003,
        "dynamiq.usage.output_cost_usd": 0.0006,
        "dynamiq.usage.total_cost_usd": 0.0009,
    }


def test_usage_with_cache_tokens(settings):
    usage = USAGE | {"cache_read_input_tokens": 100, "cache_creation_input_tokens": 20}

    attributes = call_execute_run(SpanType.CHAT, openai_llm(), {"usage_data": usage}, settings)

    assert attributes["gen_ai.usage.cache_read.input_tokens"] == 100
    assert attributes["gen_ai.usage.cache_creation.input_tokens"] == 20
    assert attributes["gen_ai.usage.cache_write.input_tokens"] == 20


def test_usage_without_costs(settings):
    usage = USAGE | {"prompt_tokens_cost_usd": None, "completion_tokens_cost_usd": None, "total_tokens_cost_usd": None}

    attributes = call_execute_run(SpanType.CHAT, openai_llm(), {"usage_data": usage}, settings)

    assert not any("cost" in key for key in attributes)
    assert attributes["gen_ai.usage.input_tokens"] == 120


def test_usage_on_a_span_that_calls_a_model_without_an_llm_node(settings):
    usage = {
        "prompt_tokens": 10,
        "completion_tokens": 0,
        "total_tokens": 10,
        "prompt_tokens_cost_usd": 0,
        "completion_tokens_cost_usd": 0.0,
        "total_tokens_cost_usd": 0,
    }

    attributes = call_execute_run(SpanType.NODE, Input(), {"usage_data": usage}, settings)

    assert attributes == {
        "dynamiq.usage.input_tokens": 10,
        "dynamiq.usage.output_tokens": 0,
        "dynamiq.usage.total_tokens": 10,
        "dynamiq.usage.input_cost_usd": 0.0,
        "dynamiq.usage.output_cost_usd": 0.0,
        "dynamiq.usage.total_cost_usd": 0.0,
    }
    assert not any(key.startswith("gen_ai.usage") for key in attributes)


def test_output_message_uses_the_reported_finish_reason(settings):
    state = {}
    call_execute_run(SpanType.CHAT, openai_llm(), {"response_data": {"finish_reasons": ["length"]}}, settings, state)

    attributes = call_execute_output(SpanType.CHAT, openai_llm(), {"content": "Partial"}, settings, state)

    assert validate_messages(attributes["gen_ai.output.messages"], "output") == [
        {"role": "assistant", "parts": [{"type": "text", "content": "Partial"}], "finish_reason": "length"}
    ]
    assert "gen_ai.response.finish_reasons" not in attributes


@pytest.mark.parametrize(
    ("output", "finish_reason"),
    [
        ({"content": "Done."}, "stop"),
        ({"content": None, "tool_calls": [{"id": "c", "function": {"name": "f", "arguments": {}}}]}, "tool_call"),
        ({"content": "Cut", "finish_reason": "length"}, "length"),
    ],
    ids=["text", "tool-calls", "from-output"],
)
def test_output_message_finish_reason_without_a_report(settings, output, finish_reason):
    attributes = call_execute_output(SpanType.CHAT, openai_llm(), output, settings)

    [message] = validate_messages(attributes["gen_ai.output.messages"], "output")
    assert message["finish_reason"] == finish_reason
    assert attributes["gen_ai.response.finish_reasons"] == [finish_reason]


def test_mocked_chat_run_gets_output_messages_from_its_output(settings):
    state = {}

    attributes = call_output(SpanType.CHAT, openai_llm(), {"content": "mocked"}, settings, {"is_mocked": True}, state)

    assert json.loads(attributes["gen_ai.output.messages"])[0]["parts"] == [{"type": "text", "content": "mocked"}]


def test_executed_chat_run_does_not_rebuild_output_messages_at_the_end(settings):
    state = {}
    call_execute_output(SpanType.CHAT, openai_llm(), {"content": "raw"}, settings, state)

    attributes = call_output(SpanType.CHAT, openai_llm(), {"content": "transformed"}, settings, state=state)

    assert "gen_ai.output.messages" not in attributes
    assert attributes["dynamiq.output"] == '{"content":"transformed"}'


# Embeddings spans


def test_document_embeddings_usage_model_and_dimensions(settings):
    output = {
        "documents": [Document(content="a", embedding=[0.1, 0.2, 0.3])],
        "meta": {"model": "text-embedding-3-small", "usage": {"prompt_tokens": 6, "total_tokens": 6}},
    }

    attributes = call_execute_output(SpanType.EMBEDDINGS, OpenAIDocumentEmbedder.model_construct(), output, settings)

    assert attributes == {
        "gen_ai.response.model": "text-embedding-3-small",
        "gen_ai.usage.input_tokens": 6,
        "dynamiq.usage.total_tokens": 6,
        "gen_ai.embeddings.dimension.count": 3,
    }


def test_text_embedder_response_data_gives_model_and_usage(settings):
    meta = {
        "model": "text-embedding-3-small",
        "usage": {"prompt_tokens": 6, "completion_tokens": 0, "total_tokens": 6, "prompt_tokens_details": None},
    }

    attributes = call_execute_run(
        SpanType.EMBEDDINGS, OpenAITextEmbedder.model_construct(), {"response_data": meta}, settings
    )

    assert attributes == {
        "gen_ai.response.model": "text-embedding-3-small",
        "gen_ai.usage.input_tokens": 6,
        "dynamiq.usage.total_tokens": 6,
    }


def test_embedding_dimensions_from_the_node_win(settings):
    node = OpenAITextEmbedder.model_construct(dimensions=256)
    state = {}

    assert (
        call_input(SpanType.EMBEDDINGS, node, {"query": "q"}, settings, state=state)[
            "gen_ai.embeddings.dimension.count"
        ]
        == 256
    )
    output = TextEmbeddingOutput(embedding=[0.0] * 1536, query="q")
    assert "gen_ai.embeddings.dimension.count" not in call_execute_output(
        SpanType.EMBEDDINGS, node, output, settings, state
    )


def test_text_embedding_output_vector_is_cut_to_the_trace_limit(settings):
    output = TextEmbeddingOutput(embedding=[0.5] * 1536, query="q")

    attributes = call_output(SpanType.EMBEDDINGS, OpenAITextEmbedder.model_construct(), output, settings)

    assert len(json.loads(attributes["dynamiq.output"])["embedding"]) == 20


# Retrieval spans


def test_retrieval_input_and_documents(settings):
    node = ChromaDocumentRetriever.model_construct(index_name="docs", top_k=10)

    attributes = call_input(
        SpanType.RETRIEVAL, node, {"embedding": [0.1], "top_k": 3, "query": "capital of France"}, settings
    )
    output = {"documents": [Document(id="d1", content="Paris", score=0.9), Document(id="d2", content="x")]}
    output_attributes = call_execute_output(SpanType.RETRIEVAL, node, output, settings)

    assert attributes["gen_ai.request.top_k"] == 3.0
    assert attributes["gen_ai.retrieval.query.text"] == "capital of France"
    assert json.loads(output_attributes["gen_ai.retrieval.documents"]) == [{"id": "d1", "score": 0.9}, {"id": "d2"}]


def test_retrieval_top_k_falls_back_to_the_node(settings):
    node = ChromaDocumentRetriever.model_construct(index_name="docs", top_k=10)

    attributes = call_input(SpanType.RETRIEVAL, node, {"embedding": [0.1], "top_k": 0}, settings)

    assert attributes["gen_ai.request.top_k"] == 10.0
    assert "gen_ai.retrieval.query.text" not in attributes


# Agent spans


def test_agent_input_and_output_messages(settings):
    agent = Agent(llm=openai_llm(), description="Answers support questions.")

    input_attributes = call_input(SpanType.AGENT, agent, {"input": "Where is my order?"}, settings)
    output_attributes = call_output(SpanType.AGENT, agent, {"content": "It ships today."}, settings)

    assert input_attributes["gen_ai.agent.description"] == "Answers support questions."
    assert validate_messages(input_attributes["gen_ai.input.messages"], "input") == [
        {"role": "user", "parts": [{"type": "text", "content": "Where is my order?"}]}
    ]
    assert validate_messages(output_attributes["gen_ai.output.messages"], "output") == [
        {"role": "assistant", "parts": [{"type": "text", "content": "It ships today."}], "finish_reason": "stop"}
    ]


def test_agent_structured_answer_is_json_text(settings):
    attributes = call_output(SpanType.AGENT, Agent(llm=openai_llm()), {"content": {"answer": 42}}, settings)

    [message] = validate_messages(attributes["gen_ai.output.messages"], "output")
    assert json.loads(message["parts"][0]["content"]) == {"answer": 42}


def test_agent_input_redacts_tool_params(settings):
    input_data = {"input": "q", "tool_params": {"global_params": {"mcp_http_headers": {"Authorization": "Bearer t0k"}}}}

    attributes = call_input(SpanType.AGENT, Agent(llm=openai_llm()), input_data, settings)

    assert "t0k" not in attributes["dynamiq.input"]


# Workflow spans


def test_workflow_input_and_output_of_the_single_terminal_node(settings):
    first = Input()
    last = Output(depends=[NodeDependency(first)])
    workflow = Workflow(flow=Flow(nodes=[first, last]))
    output = {
        first.id: {"status": "success", "input": {"q": 1}, "output": {"q": 1}},
        last.id: {"status": "success", "input": {"q": 1}, "output": {"answer": "done"}},
    }

    assert call_input(SpanType.WORKFLOW, workflow, {"q": 1}, settings) == {"dynamiq.input": '{"q":1}'}
    assert call_output(SpanType.WORKFLOW, workflow, output, settings) == {"dynamiq.output": '{"answer":"done"}'}


def test_workflow_output_of_several_terminal_nodes_is_keyed_by_node(settings):
    first, second, root = Output(), Output(), Input()
    first.depends = [NodeDependency(root)]
    second.depends = [NodeDependency(root)]
    flow = Flow(nodes=[root, first, second])
    output = {
        root.id: {"status": "success", "output": "r"},
        first.id: {"status": "success", "output": "a"},
        second.id: {"status": "success", "output": "b"},
    }

    attributes = call_output(SpanType.WORKFLOW, flow, output, settings)

    assert json.loads(attributes["dynamiq.output"]) == {first.id: "a", second.id: "b"}


def test_failed_nodes(settings):
    failed = [RunnableFailedNodeInfo(id="n1", name="fetch", error_message="timeout talking to api")]

    attributes = checked(semconv.failure_attributes({"failed_nodes": failed}, settings=settings))

    assert json.loads(attributes["dynamiq.failed_nodes"]) == [
        {"id": "n1", "name": "fetch", "error_message": "timeout talking to api"}
    ]
    assert semconv.failure_attributes({"failed_nodes": []}, settings=settings) == {}


def test_resume_attributes():
    run_kwargs = {"resumed_from_checkpoint": "cp-1", "original_run_id": "run-0"}

    assert checked(semconv.resume_attributes(run_kwargs)) == {
        "dynamiq.resume.checkpoint_id": "cp-1",
        "dynamiq.resume.original_run_id": "run-0",
    }
    assert semconv.resume_attributes({}) == {}


def test_flow_run_and_pause_attributes():
    flow = Flow(nodes=[Input()])
    workflow = Workflow(flow=flow)

    class Checkpoint:
        id = "cp-9"

    flow._checkpoint = Checkpoint()

    assert semconv.flow_run_attributes("flow-run") == {"dynamiq.flow.run_id": "flow-run"}
    assert semconv.pause_attributes(workflow) == {"dynamiq.checkpoint.id": "cp-9"}
    assert semconv.pause_attributes(flow) == {"dynamiq.checkpoint.id": "cp-9"}


# Events and errors


def test_hook_events(settings):
    events = [
        {
            "hook": "pii-guard",
            "type": "Guard",
            "point": "before_tool",
            "decision": "ask",
            "tool": "send",
            "outcome": "ok",
        }
    ]

    [(name, attributes)] = semconv.hook_events({"hook_events": events}, settings=settings)

    assert name == "dynamiq.agent.hook"
    assert checked(attributes) == {
        "dynamiq.hook.name": "pii-guard",
        "dynamiq.hook.type": "Guard",
        "dynamiq.hook.point": "before_tool",
        "dynamiq.hook.decision": "ask",
        "dynamiq.hook.details": '{"tool":"send","outcome":"ok"}',
    }


def test_execution_events():
    settings = CaptureSettings()

    succeeded = semconv.execution_event(
        execution_id="e2", attempt=2, succeeded=True, start_time_unix_nano=10, error=None, settings=settings
    )
    failed = semconv.execution_event(
        execution_id="e1",
        attempt=1,
        succeeded=False,
        start_time_unix_nano=5,
        error=TimeoutError("slow"),
        settings=settings,
    )

    assert checked(succeeded) == {
        "dynamiq.execution.id": "e2",
        "dynamiq.execution.attempt": 2,
        "dynamiq.execution.status": "succeeded",
        "dynamiq.execution.start_time_unix_nano": 10,
    }
    assert checked(failed) == {
        "dynamiq.execution.id": "e1",
        "dynamiq.execution.attempt": 1,
        "dynamiq.execution.status": "failed",
        "dynamiq.execution.start_time_unix_nano": 5,
        "error.type": "TimeoutError",
        "exception.message": "slow",
    }


class RateLimitError(Exception):
    pass


def test_error_spec_of_a_failure(settings):
    try:
        raise RateLimitError("429 Too Many Requests")
    except RateLimitError as error:
        spec = semconv.error_spec(error, RunOutcome.FAILED, settings=settings)

    assert spec.error_type == "RateLimitError"
    assert spec.description == "429 Too Many Requests"
    assert spec.event_attributes["exception.type"] == "RateLimitError"
    assert spec.event_attributes["exception.message"] == "429 Too Many Requests"
    assert "raise RateLimitError" in spec.event_attributes["exception.stacktrace"]
    checked(spec.event_attributes)


def test_error_spec_of_a_cancel_and_a_pause(settings):
    canceled = semconv.error_spec(None, RunOutcome.CANCELED, settings=settings)

    assert canceled.error_type == "CanceledException"
    assert canceled.event_attributes is None
    assert semconv.error_spec(RunPausedException("paused"), RunOutcome.PAUSED, settings=settings) is None
    assert semconv.error_spec(None, RunOutcome.SUCCEEDED, settings=settings) is None


# Content off: no content attribute at all


def test_no_content_attribute_without_content_capture(no_content):
    """Every Content-level attribute of the spec, with content capture off."""
    llm = openai_llm(tools=[SEARCH_TOOL])
    agent = Agent(llm=openai_llm())
    retriever = ChromaDocumentRetriever.model_construct(index_name="docs")
    workflow = Workflow(flow=Flow(nodes=[Input()]))
    secret = "user-secret-text"
    settings = no_content.model_copy(update={"node_config_enabled": True})
    collected: dict[str, Any] = {}

    for span_type, entity in [
        (SpanType.NODE, Input()),
        (SpanType.TOOL, ToolNode()),
        (SpanType.CHAT, llm),
        (SpanType.AGENT, agent),
        (SpanType.RETRIEVAL, retriever),
        (SpanType.WORKFLOW, workflow),
    ]:
        input_data = {"input": secret, "query": secret, "embedding": [0.1]}
        collected |= call_input(span_type, entity, input_data, settings)
        collected |= call_output(span_type, entity, {"content": secret}, settings)
        collected |= call_execute_output(
            span_type, entity, {"content": secret, "documents": [Document(id="d", content=secret)]}, settings
        )
    collected |= call_execute_run(
        SpanType.CHAT, llm, {"prompt_messages": [{"role": "user", "content": secret}], "usage_data": USAGE}, settings
    )
    collected |= checked(
        semconv.skip_attributes(
            {"failed_dependency": {}},
            {"q": secret},
            secret,
            span_type=SpanType.NODE,
            entity=Input(),
            settings=settings,
            state={},
        )
    )
    collected |= checked(
        semconv.failure_attributes(
            {"failed_nodes": [RunnableFailedNodeInfo(id="n", name="x", error_message=secret)]}, settings=settings
        )
    )
    [(_, hook)] = semconv.hook_events(
        {"hook_events": [{"hook": "h", "decision": "allow", "x": secret}]}, settings=settings
    )
    collected |= hook
    error = ValueError(secret)
    spec = semconv.error_spec(error, RunOutcome.FAILED, settings=settings)
    execution = semconv.execution_event(
        execution_id="e", attempt=1, succeeded=False, start_time_unix_nano=1, error=error, settings=settings
    )

    content_keys = {
        "dynamiq.input",
        "dynamiq.output",
        "dynamiq.node.config",
        "dynamiq.human_feedback",
        "dynamiq.hook.details",
        "gen_ai.input.messages",
        "gen_ai.output.messages",
        "gen_ai.tool.definitions",
        "gen_ai.tool.call.arguments",
        "gen_ai.tool.call.result",
        "gen_ai.retrieval.query.text",
        "gen_ai.retrieval.documents",
        "exception.message",
        "exception.stacktrace",
    }
    assert not content_keys & collected.keys()
    assert secret not in json.dumps(collected)
    # Non-content attributes stay.
    assert collected["gen_ai.usage.input_tokens"] == 120
    assert json.loads(collected["dynamiq.failed_nodes"]) == [{"id": "n", "name": "x"}]
    assert collected["dynamiq.hook.decision"] == "allow"
    assert spec.error_type == "ValueError"
    assert spec.description is None
    assert spec.event_attributes == {"exception.type": "ValueError"}
    assert not content_keys & execution.keys()
    assert execution["error.type"] == "ValueError"
