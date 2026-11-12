"""The run-based TracingCallbackHandler records the same runs whether or not spans are emitted."""

import pytest

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes.node import ErrorHandling
from dynamiq.runnables import RunnableConfig
from dynamiq.types.telemetry import TracingConfig
from tests.unit.telemetry.conftest import Echo, Failing, Flaky

pytest.importorskip("opentelemetry.sdk")

pytestmark = pytest.mark.usefixtures("no_span_leaks")


def workflow_with_every_outcome() -> Workflow:
    """A step that succeeds, one that needs a retry, one that fails and one that is skipped behind it."""
    first = Echo(id="first", name="first")
    retried = Flaky(
        id="retried", name="retried", error_handling=ErrorHandling(max_retries=1, retry_interval_seconds=0.01)
    )
    retried.depends_on(first)
    failing = Failing(id="failing", name="failing")
    failing.depends_on(retried)
    skipped = Echo(id="skipped", name="skipped")
    skipped.depends_on(failing)
    return Workflow(id="wf", flow=Flow(id="flow", nodes=[first, retried, failing, skipped]))


def recorded_runs(handler: TracingCallbackHandler) -> list[dict]:
    """The handler's runs without what differs between any two runs: ids, times and tracebacks."""
    position = {run_id: index for index, run_id in enumerate(handler.runs)}
    runs = []
    for run in handler.runs.values():
        runs.append(
            {
                "name": run.name,
                "type": run.type,
                "status": run.status,
                "parent": position.get(run.parent_run_id),
                "input": run.input,
                "output": run.output,
                "error": {key: value for key, value in (run.error or {}).items() if key != "traceback"},
                "metadata": run.metadata,
                "tags": run.tags,
                "executions": [
                    {
                        "status": execution.status,
                        "input": execution.input,
                        "output": execution.output,
                        "error": {key: value for key, value in (execution.error or {}).items() if key != "traceback"},
                        "metadata": execution.metadata,
                    }
                    for execution in run.executions
                ],
            }
        )
    return runs


def run_with_handler(config_fields: dict) -> tuple[list[dict], dict]:
    handler = TracingCallbackHandler(trace_id="trace", source_id="source", session_id="session")
    config = RunnableConfig(run_id="config-run", callbacks=[handler], **config_fields)
    result = workflow_with_every_outcome().run(input_data={"value": 1}, config=config)
    return recorded_runs(handler), result.to_dict()


def test_handler_records_the_same_runs_with_and_without_a_provider(exporter, provider):
    without_provider, result_without = run_with_handler({})
    assert exporter.get_finished_spans() == ()

    with_provider, result_with = run_with_handler({"tracing": TracingConfig(tracer_provider=provider)})

    assert len(exporter.get_finished_spans()) == 5
    assert [run["name"] for run in with_provider] == ["Workflow", "Flow", "first", "retried", "failing", "skipped"]
    assert with_provider == without_provider
    assert result_with == result_without
