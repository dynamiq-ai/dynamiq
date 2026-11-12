"""Input -> LLM -> Output through ``Workflow.run``: the span tree, names, kinds and the chat attributes."""

import json

import pytest
from opentelemetry.trace import SpanKind, StatusCode

from dynamiq import Workflow
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.flows import Flow
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.node import NodeDependency
from dynamiq.nodes.utils import Input, Output
from dynamiq.prompts import Message, MessageRole, Prompt
from dynamiq.runnables import RunnableStatus
from dynamiq.types.streaming import StreamingConfig
from tests.integration.telemetry.conftest import OPERATION, RUN_ID, RUN_STATUS, RUN_TYPE, attrs_json


def build_workflow(**llm_kwargs) -> Workflow:
    start = Input(id="in", name="start")
    llm = OpenAI(
        id="llm",
        name="answerer",
        connection=OpenAIConnection(api_key="test-key"),
        model="gpt-4o",
        prompt=Prompt(
            messages=[
                Message(role=MessageRole.SYSTEM, content="Be brief."),
                Message(role=MessageRole.USER, content="Say {{ topic }}"),
            ]
        ),
        depends=[NodeDependency(start)],
        **llm_kwargs,
    )
    end = Output(id="out", name="end", depends=[NodeDependency(llm)])
    return Workflow(id="wf-1", version="3", flow=Flow(id="flow-1", nodes=[start, llm, end]))


def test_tree_names_and_kinds(mock_llm_executor, config, trace):
    workflow = build_workflow()

    result = workflow.run(input_data={"topic": "hello"}, config=config)

    assert result.status == RunnableStatus.SUCCESS
    spans = trace()
    assert len(spans.spans) == 4  # the flow of a workflow has no span of its own
    root = spans.one("invoke_workflow Workflow")
    assert root.parent is None
    assert root.kind == SpanKind.INTERNAL
    assert [s.name for s in spans.children(root)] == ["start", "chat gpt-4o", "end"]
    assert {s.context.trace_id for s in spans.spans} == {root.context.trace_id}
    assert spans.one("chat gpt-4o").kind == SpanKind.CLIENT
    assert spans.one("start").kind == SpanKind.INTERNAL
    assert OPERATION not in spans.one("start").attributes
    assert OPERATION not in spans.one("end").attributes


def test_every_span_carries_run_identity_and_ends_with_a_status(mock_llm_executor, config, trace):
    build_workflow().run(input_data={"topic": "hello"}, config=config)

    spans = trace().spans
    run_ids = [s.attributes[RUN_ID] for s in spans]
    assert len(set(run_ids)) == len(spans)
    assert {s.attributes[RUN_TYPE] for s in spans} == {"workflow", "node"}
    for span in spans:
        assert span.attributes[RUN_STATUS] == "succeeded"
        assert span.status.status_code == StatusCode.UNSET
        assert span.end_time is not None and span.end_time >= span.start_time


def test_workflow_span_attributes(mock_llm_executor, config, make_config, trace):
    workflow = build_workflow()
    config = make_config(session_id="sess-1", user_id="user-1", source_id="src-1", tags=["a", "b"], metadata={"k": 1})
    config.run_id = "cfg-run-1"

    result = workflow.run(input_data={"topic": "hello"}, config=config)

    span = trace().one("invoke_workflow Workflow")
    attributes = span.attributes
    assert attributes[OPERATION] == "invoke_workflow"
    assert attributes["gen_ai.workflow.name"] == "Workflow"
    assert attributes["dynamiq.workflow.id"] == "wf-1"
    assert attributes["dynamiq.workflow.version"] == "3"
    assert attributes["dynamiq.flow.id"] == "flow-1"
    assert attributes["dynamiq.flow.run_id"]
    assert attributes["dynamiq.config.run_id"] == "cfg-run-1"
    assert attrs_json(span, "dynamiq.input") == {"topic": "hello"}
    # one terminal node (nothing depends on "out"): the output itself, not keyed by node id
    assert attrs_json(span, "dynamiq.output") == json.loads(json.dumps(result.output["out"]["output"]))
    # run-level values are repeated on every span of the run
    for s in trace().spans:
        assert s.attributes["gen_ai.conversation.id"] == "sess-1"
        assert s.attributes["session.id"] == "sess-1"
        assert s.attributes["user.id"] == "user-1"
        assert s.attributes["dynamiq.source.id"] == "src-1"
        assert list(s.attributes["dynamiq.tags"]) == ["a", "b"]
        assert s.attributes["dynamiq.metadata.k"] == 1


def test_chat_span_attributes(mock_llm_executor, config, trace):
    workflow = build_workflow(
        temperature=0.25, max_tokens=64, top_p=0.9, stop=["END"], seed=7, presence_penalty=0.1, frequency_penalty=0.2
    )

    workflow.run(input_data={"topic": "hello"}, config=config)

    span = trace().one("chat gpt-4o")
    a = span.attributes
    assert a[OPERATION] == "chat"
    assert a["gen_ai.provider.name"] == "openai"
    assert a["gen_ai.system"] == "openai"
    assert a["gen_ai.request.model"] == "gpt-4o"
    assert a["gen_ai.request.temperature"] == 0.25
    assert a["gen_ai.request.max_tokens"] == 64
    assert a["gen_ai.request.top_p"] == 0.9
    assert a["gen_ai.request.frequency_penalty"] == 0.2
    assert a["gen_ai.request.presence_penalty"] == 0.1
    assert list(a["gen_ai.request.stop_sequences"]) == ["END"]
    assert a["gen_ai.request.seed"] == 7
    assert "gen_ai.request.stream" not in a
    # the model as configured keeps its provider prefix, the request model does not
    assert a["dynamiq.node.model"] == "openai/gpt-4o"
    assert a["dynamiq.node.id"] == "llm"
    assert a["dynamiq.node.name"] == "answerer"
    assert a["dynamiq.node.type"] == "dynamiq.nodes.llms.OpenAI"
    assert a["dynamiq.node.group"] == "llms"
    assert list(a["dynamiq.node.depends"]) == ["in"]
    assert a["gen_ai.response.id"]
    assert list(a["gen_ai.response.finish_reasons"]) == ["stop"]
    assert "gen_ai.system_instructions" not in a
    assert "gen_ai.tool.call.id" not in a


def test_chat_span_leaves_out_request_parameters_that_are_not_set(mock_llm_executor, config, trace):
    build_workflow().run(input_data={"topic": "hello"}, config=config)

    attributes = trace().one("chat gpt-4o").attributes
    for key in (
        "gen_ai.request.top_p",
        "gen_ai.request.seed",
        "gen_ai.request.stop_sequences",
        "gen_ai.request.presence_penalty",
        "gen_ai.request.frequency_penalty",
        "gen_ai.output.type",
    ):
        assert key not in attributes


def test_chat_messages_follow_the_message_schema(mock_llm_executor, config, trace):
    build_workflow().run(input_data={"topic": "hello"}, config=config)

    span = trace().one("chat gpt-4o")
    assert attrs_json(span, "gen_ai.input.messages") == [
        {"role": "system", "parts": [{"type": "text", "content": "Be brief."}]},
        {"role": "user", "parts": [{"type": "text", "content": "Say hello"}]},
    ]
    assert attrs_json(span, "gen_ai.output.messages") == [
        {"role": "assistant", "parts": [{"type": "text", "content": "mocked_response"}], "finish_reason": "stop"}
    ]
    # dynamiq.input is the node input (template parameters), not the messages
    assert attrs_json(span, "dynamiq.input")["topic"] == "hello"
    assert attrs_json(span, "dynamiq.output") == {"content": "mocked_response"}


def test_chat_usage_and_cost(script_llm, config, trace):
    from dynamiq.nodes.types import InferenceMode
    from tests.helpers.agent_hooks import final_step

    script_llm(InferenceMode.DEFAULT, [final_step("ok")], usage=(11, 7))

    build_workflow().run(input_data={"topic": "hello"}, config=config)

    a = trace().one("chat gpt-4o").attributes
    assert a["gen_ai.usage.input_tokens"] == 11
    assert a["gen_ai.usage.output_tokens"] == 7
    assert a["dynamiq.usage.total_tokens"] == 18
    assert a["dynamiq.usage.input_cost_usd"] > 0
    assert a["dynamiq.usage.output_cost_usd"] > 0
    assert a["dynamiq.usage.total_cost_usd"] == pytest.approx(
        a["dynamiq.usage.input_cost_usd"] + a["dynamiq.usage.output_cost_usd"]
    )
    assert "dynamiq.usage.input_tokens" not in a  # reserved for spans other than chat and embeddings
    # tokens are never recorded on the workflow or the other nodes: a backend summing spans would count them twice
    for span in trace().spans:
        if span.name != "chat gpt-4o":
            assert not [key for key in span.attributes if "usage" in key]


def test_streaming_llm(mock_llm_executor, config, trace):
    workflow = build_workflow(streaming=StreamingConfig(enabled=True))

    result = workflow.run(input_data={"topic": "hello"}, config=config)

    assert result.status == RunnableStatus.SUCCESS
    span = trace().one("chat gpt-4o")
    assert span.attributes["gen_ai.request.stream"] is True
    assert attrs_json(span, "gen_ai.output.messages")[0]["parts"] == [{"type": "text", "content": "mocked_response"}]
    assert span.attributes[RUN_STATUS] == "succeeded"
    assert [s.name for s in trace().children(trace().one("invoke_workflow Workflow"))] == [
        "start",
        "chat gpt-4o",
        "end",
    ]


def test_async_run_produces_the_same_tree(mock_llm_executor, config, trace):
    import asyncio

    workflow = build_workflow()

    result = asyncio.run(workflow.run_async(input_data={"topic": "hello"}, config=config))

    assert result.status == RunnableStatus.SUCCESS
    spans = trace()
    assert spans.shape() == [
        "invoke_workflow Workflow",
        sorted([["start", []], ["chat gpt-4o", []], ["end", []]], key=repr),
    ]


def test_workflow_output_is_keyed_by_node_id_when_there_are_several_terminal_nodes(config, trace):
    from dynamiq.nodes.operators.operators import Pass

    start = Input(id="in", name="start")
    left = Pass(id="left", name="left", depends=[NodeDependency(start)])
    right = Pass(id="right", name="right", depends=[NodeDependency(start)])
    workflow = Workflow(flow=Flow(nodes=[start, left, right]))

    result = workflow.run(input_data={"v": 1}, config=config)

    output = attrs_json(trace().one("invoke_workflow Workflow"), "dynamiq.output")
    assert set(output) == {"left", "right"}  # "in" has dependents, so it is not terminal
    assert output["left"] == json.loads(json.dumps(result.output["left"]["output"]))
    assert output["right"] == json.loads(json.dumps(result.output["right"]["output"]))
