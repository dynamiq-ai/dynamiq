"""Protects a backend projection: runs rebuilt from spans equal what a ``TracingCallbackHandler`` recorded."""

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.callbacks.tracing import RunType
from dynamiq.flows import Flow
from dynamiq.nodes import ErrorHandling
from dynamiq.nodes.agents.hooks import CallLimitHook
from dynamiq.nodes.node import NodeDependency
from dynamiq.nodes.operators.operators import Pass
from dynamiq.nodes.types import Behavior, InferenceMode
from dynamiq.runnables import RunnableStatus
from dynamiq.telemetry.rebuild import RebuiltRun, rebuild_runs
from tests.helpers.agent_hooks import ExplodingTool, SearchTool, build_agent, final_step, tool_step
from tests.integration.telemetry.equivalence import ScenarioOptions, assert_equivalent, to_json
from tests.integration.telemetry.golden_scenarios import Failing, Flaky

TAGS = ["rebuild", "e2e"]


def assert_matches(rebuilt: dict[str, RebuiltRun], handler: TracingCallbackHandler, spans: list) -> None:
    """Every field of every rebuilt run agrees with the handler's run, under the documented equivalence."""
    assert_equivalent(
        [run.model_dump() for run in rebuilt.values()],
        [run.to_dict() for run in handler.runs.values()],
        spans,
        ScenarioOptions(),
    )


def rebuilt_runs(spans: list) -> list[dict]:
    return to_json(rebuild_runs(spans))


def make_handler() -> TracingCallbackHandler:
    return TracingCallbackHandler(tags=TAGS, source_id="source-1", session_id="session-1")


def traced(make_config, handler):
    return make_config(callbacks=[handler], tags=TAGS, source_id="source-1", session_id="session-1")


def test_agent_trace_rebuilds_into_the_runs_the_handler_recorded(script_llm, make_config, trace):
    handler = make_handler()
    mode = InferenceMode.FUNCTION_CALLING
    agent = build_agent(
        tools=[SearchTool(result="forty-two"), ExplodingTool()],
        inference_mode=mode,
        hooks=[CallLimitHook(tools=["search"], max_per_run=1)],
        description="Answers questions.",
    )
    flaky = Flaky(id="flaky", error_handling=ErrorHandling(max_retries=2, retry_interval_seconds=0.01))
    workflow = Workflow(id="wf", version="2", flow=Flow(id="flow", nodes=[agent, flaky]))
    steps = [
        tool_step("search", query="q1"),
        tool_step("search", query="q2"),  # over the limit: the hook blocks it
        tool_step("http", query="x"),  # fails, the agent recovers
        final_step("done"),
    ]
    script_llm(mode, steps)

    result = workflow.run(input_data={"input": "hello"}, config=traced(make_config, handler))

    assert result.status == RunnableStatus.SUCCESS
    spans = trace().spans
    handler_runs = list(handler.runs.values())
    # the scenario exercises what the table covers: a retry, a failed tool, a blocked tool, usage, a hook event
    assert any(len(r.executions) == 2 for r in handler_runs)
    assert any(r.status.value == "failed" for r in handler_runs if r.type == RunType.NODE)
    assert any(r.metadata.get("hooks") for r in handler_runs if r.type == RunType.NODE)
    assert sum(1 for r in handler_runs if r.metadata.get("usage")) == 4  # one per model call
    assert_equivalent(rebuild_runs(spans), [run.to_dict() for run in handler_runs], spans, ScenarioOptions())


def test_workflow_output_is_the_map_of_the_nodes_it_ran(make_config, trace):
    handler = make_handler()
    first = Pass(id="first", name="first")
    second = Pass(id="second", name="second", depends=[NodeDependency(first)])

    Workflow(flow=Flow(nodes=[first, second])).run(input_data={"v": 1}, config=traced(make_config, handler))

    (workflow_run,) = (run for run in rebuilt_runs(trace().spans) if run["type"] == "workflow")
    (recorded,) = to_json([run.to_dict() for run in handler.runs.values() if run.type == RunType.WORKFLOW])
    assert workflow_run["output"] == recorded["output"]
    assert set(workflow_run["output"]) == {"first", "second"}


def test_workflow_output_keeps_the_type_and_message_of_a_node_that_failed_and_returned(make_config, trace):
    handler = make_handler()
    soft = Failing(id="soft", error_handling=ErrorHandling(behavior=Behavior.RETURN))
    after = Pass(id="after", name="after", depends=[NodeDependency(soft)])

    result = Workflow(flow=Flow(nodes=[soft, after])).run(input_data={"v": 1}, config=traced(make_config, handler))

    assert result.status == RunnableStatus.SUCCESS
    (workflow_run,) = (run for run in rebuilt_runs(trace().spans) if run["type"] == "workflow")
    (recorded,) = to_json([run.to_dict() for run in handler.runs.values() if run.type == RunType.WORKFLOW])
    assert recorded["output"]["soft"]["error"]["type"] == "RuntimeError"
    assert workflow_run["output"]["soft"] == {
        "status": "failure",
        "input": recorded["output"]["soft"]["input"],
        "output": None,
        "error": {"type": "RuntimeError", "message": "the node broke"},
    }
    assert workflow_run["output"]["after"] == recorded["output"]["after"]


def test_failed_workflow_rebuilds_with_failed_nodes_and_without_output(make_config, trace):
    handler = make_handler()
    broken = Failing(id="broken", name="broken")
    after = Pass(id="after", name="after", depends=[NodeDependency(broken)])

    result = Workflow(id="wf", flow=Flow(id="flow", nodes=[broken, after])).run(
        input_data={"v": 1}, config=traced(make_config, handler)
    )

    assert result.status == RunnableStatus.FAILURE
    spans = trace().spans
    rebuilt = rebuilt_runs(spans)
    assert_equivalent(rebuilt, [run.to_dict() for run in handler.runs.values()], spans, ScenarioOptions())
    (workflow_run,) = (run for run in rebuilt if run["type"] == "workflow")
    assert workflow_run["output"] is None
    assert workflow_run["metadata"]["failed_nodes"] == [
        {"id": "broken", "name": "broken", "error_message": "the node broke"}
    ]
    (skipped,) = (run for run in rebuilt if run["name"] == "after")
    assert skipped["status"] == "skipped"
    assert skipped["executions"] == []
    assert skipped["metadata"]["skip"]["failed_dependency"]["node"]["id"] == "broken"


def test_rebuilt_parent_links_form_the_run_tree(script_llm, make_config, trace):
    handler = make_handler()
    mode = InferenceMode.XML
    agent = build_agent(tools=[SearchTool()], inference_mode=mode)
    script_llm(mode, [tool_step("search", query="q"), final_step("ok")])

    agent.run(input_data={"input": "hello"}, config=traced(make_config, handler))

    rebuilt = rebuilt_runs(trace().spans)
    roots = [run for run in rebuilt if run["parent_run_id"] is None]
    assert [run["name"] for run in roots] == ["assistant"]
    assert sorted(run["name"] for run in rebuilt if run["parent_run_id"] == roots[0]["id"]) == ["LLM", "LLM", "search"]
    recorded_parents = {
        str(run.id): str(run.parent_run_id) if run.parent_run_id else None for run in handler.runs.values()
    }
    assert {run["id"]: run["parent_run_id"] for run in rebuilt} == recorded_parents
