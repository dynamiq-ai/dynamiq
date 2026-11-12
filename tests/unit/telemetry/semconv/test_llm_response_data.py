from typing import Any

from dynamiq import connections
from dynamiq.callbacks import BaseCallbackHandler
from dynamiq.nodes.llms import OpenAI
from dynamiq.prompts import Message, Prompt
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.telemetry import semconv
from dynamiq.telemetry.types import CaptureSettings, SpanType


class ExecuteRunRecorder(BaseCallbackHandler):
    def __init__(self):
        self.reports: list[dict[str, Any]] = []

    def on_node_execute_run(self, serialized: dict[str, Any], **kwargs: Any):
        self.reports.append(kwargs)


def test_llm_reports_response_facts_with_usage(mock_llm_executor):
    llm = OpenAI(
        model="gpt-4o",
        connection=connections.OpenAI(api_key="test-key"),
        prompt=Prompt(messages=[Message(role="user", content="Hi")]),
    )
    recorder = ExecuteRunRecorder()

    result = llm.run(input_data={}, config=RunnableConfig(callbacks=[recorder]))

    assert result.status == RunnableStatus.SUCCESS
    [report] = [report for report in recorder.reports if "usage_data" in report]
    response_data = report["response_data"]
    assert isinstance(response_data["id"], str) and response_data["id"]
    assert response_data["finish_reasons"] == ["stop"]

    state = {}
    attributes = semconv.execute_run_attributes(
        SpanType.CHAT, llm, run_kwargs=report, settings=CaptureSettings(), state=state
    )
    assert attributes["gen_ai.response.id"] == response_data["id"]
    assert attributes["gen_ai.response.finish_reasons"] == ["stop"]
