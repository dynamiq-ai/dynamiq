"""Regressions of the span mapping: binary content, retrieval documents, skips, embedder and LLM response facts."""

import asyncio
import base64
import json
from typing import Any

import pytest
from litellm import ModelResponse
from litellm.types.utils import EmbeddingResponse, Usage
from pydantic import BaseModel

from dynamiq import Workflow, connections
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.embedders import OpenAITextEmbedder
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.node import NodeDependency
from dynamiq.nodes.operators.operators import ChoiceCondition, ConditionOperator, Pass
from dynamiq.nodes.utils import Input
from dynamiq.prompts import Message, Prompt
from dynamiq.runnables import RunnableResult, RunnableStatus
from dynamiq.types.telemetry import ContentCaptureMode
from tests.integration.telemetry.conftest import RUN_STATUS, all_strings, attrs_json

BLOB = b"\x89PNG" + b"PIXELS" * 5000
BLOB_B64 = base64.b64encode(BLOB).decode()
PLACEHOLDER = f"<binary omitted: {len(BLOB)} bytes>"
CONDITION_VALUE = "zebra-condition-value-7731"


class Attachment(BaseModel):
    name: str
    data: bytes


class Fetch(Node):
    group: NodeGroup = NodeGroup.TOOLS
    name: str = "fetch"

    def execute(self, input_data: dict[str, Any], config=None, **kwargs):
        return {
            "attachment": Attachment(name="a.png", data=BLOB),
            "result": RunnableResult(status=RunnableStatus.SUCCESS, output={"file": BLOB}),
        }


def test_bytes_inside_models_in_a_tool_result_are_left_out(config, trace):
    result = Fetch().run(input_data={}, config=config)

    assert result.status == RunnableStatus.SUCCESS
    span = trace().one("execute_tool fetch")
    value = span.attributes["gen_ai.tool.call.result"]
    assert BLOB_B64[:40] not in value
    assert len(value) < 1000
    decoded = json.loads(value)
    assert decoded["attachment"] == {"name": "a.png", "data": PLACEHOLDER}
    assert decoded["result"]["output"] == {"file": PLACEHOLDER}


def condition_skip_workflow() -> Workflow:
    start = Input(id="input", name="input")
    condition = ChoiceCondition(
        variable="$.output.flag", operator=ConditionOperator.STRING_EQUALS, value=CONDITION_VALUE
    )
    gated = Pass(id="gated", name="gated", depends=[NodeDependency(start, condition=condition)])
    return Workflow(flow=Flow(nodes=[start, gated]))


def test_skip_without_content_keeps_only_the_dependency_identifiers(make_config, trace):
    workflow = condition_skip_workflow()
    config = make_config(content_capture=ContentCaptureMode.NO_CONTENT)

    result = workflow.run(input_data={"flag": "other"}, config=config)

    assert result.status == RunnableStatus.SUCCESS
    span = trace().one("gated")
    assert span.attributes[RUN_STATUS] == "skipped"
    assert not any(CONDITION_VALUE in text for text in all_strings(span))
    assert attrs_json(span, "dynamiq.skip") == {
        "failed_dependency": {
            "node": {"id": "input", "name": "input", "type": "dynamiq.nodes.utils.Input"},
            "option": None,
        }
    }


def test_skip_with_content_keeps_the_dependency_condition(config, trace):
    workflow = condition_skip_workflow()

    workflow.run(input_data={"flag": "other"}, config=config)

    skip = attrs_json(trace().one("gated"), "dynamiq.skip")
    assert skip["failed_dependency"]["condition"]["value"] == CONDITION_VALUE
    assert skip["failed_dependency"]["trigger"] == "success"


@pytest.fixture
def embedding_response(mocker):
    response = EmbeddingResponse(
        model="text-embedding-3-small-2024",
        data=[{"embedding": [0.1, 0.2, 0.3], "index": 0, "object": "embedding"}],
        usage=Usage(prompt_tokens=6, total_tokens=6),
    )
    mocker.patch("dynamiq.components.embedders.base.BaseEmbedder._embedding", return_value=response)

    async def aembedding(*args, **kwargs):
        return response

    mocker.patch("dynamiq.components.embedders.base.BaseEmbedder._aembedding", side_effect=aembedding)
    return response


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_text_embedder_span_has_the_response_model_and_usage(make_config, trace, embedding_response, is_async):
    embedder = OpenAITextEmbedder(connection=connections.OpenAI(api_key="test-key"), model="text-embedding-3-small")
    handler = TracingCallbackHandler()
    config = make_config(callbacks=[handler])

    if is_async:
        result = asyncio.run(embedder.run_async(input_data={"query": "pizza"}, config=config))
    else:
        result = embedder.run_sync(input_data={"query": "pizza"}, config=config)

    assert result.status == RunnableStatus.SUCCESS
    assert dict(result.output) == {"embedding": [0.1, 0.2, 0.3], "query": "pizza"}
    span = trace().one("embeddings text-embedding-3-small")
    assert span.attributes["gen_ai.response.model"] == "text-embedding-3-small-2024"
    assert span.attributes["gen_ai.usage.input_tokens"] == 6
    assert span.attributes["dynamiq.usage.total_tokens"] == 6
    assert "gen_ai.usage.output_tokens" not in span.attributes
    [run] = handler.runs.values()
    assert "usage" not in run.metadata


def test_every_finish_reason_of_a_multi_choice_response_is_recorded(mocker, config, trace):
    response = ModelResponse(
        choices=[
            {"index": 0, "message": {"role": "assistant", "content": "one"}, "finish_reason": "stop"},
            {"index": 1, "message": {"role": "assistant", "content": "tw"}, "finish_reason": "length"},
        ],
        usage=Usage(prompt_tokens=3, completion_tokens=4, total_tokens=7),
    )
    mocker.patch("dynamiq.nodes.llms.base.BaseLLM._completion", return_value=response)
    llm = OpenAI(
        model="gpt-4o",
        connection=connections.OpenAI(api_key="test-key"),
        prompt=Prompt(messages=[Message(role="user", content="Hi")]),
        n=2,
    )

    result = llm.run(input_data={}, config=config)

    assert result.status == RunnableStatus.SUCCESS
    span = trace().one("chat gpt-4o")
    assert list(span.attributes["gen_ai.response.finish_reasons"]) == ["stop", "length"]
    [message] = attrs_json(span, "gen_ai.output.messages")
    assert message["finish_reason"] == "stop"
