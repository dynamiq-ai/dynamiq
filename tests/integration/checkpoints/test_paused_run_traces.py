"""A run that pauses is traced as paused, not failed.

A pause is how a run waits for a person or a time: it ends the process but not the run, so its trace must not
count as a failure. The nodes it waits on are paused too; the nodes that finished keep their own status, and a run
that cannot pause, because checkpointing is off, still fails.
"""

from datetime import timedelta
from queue import Queue

from litellm import ModelResponse

from dynamiq import Workflow, connections, flows
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.callbacks.tracing import RunStatus, RunType
from dynamiq.checkpoints.backends.in_memory import InMemory
from dynamiq.checkpoints.config import CheckpointBehavior, CheckpointConfig
from dynamiq.checkpoints.types import RunPausedException
from dynamiq.nodes import llms
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.operators import Delay
from dynamiq.nodes.tools import Python
from dynamiq.nodes.tools.human_feedback import HumanFeedbackAction, HumanFeedbackTool
from dynamiq.nodes.utils import Input, Output
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types.feedback import FeedbackMethod
from dynamiq.types.streaming import StreamingConfig

INPUT_TIMEOUT_SECONDS = 0.3


def runtime_checkpoint() -> CheckpointConfig:
    """Checkpointing as the platform runtime runs it: one checkpoint, saved when the run pauses."""
    return CheckpointConfig(
        enabled=True,
        backend=InMemory(),
        behavior=CheckpointBehavior.REPLACE,
        checkpoint_on_start_enabled=False,
        checkpoint_after_node_enabled=False,
        checkpoint_on_failure_enabled=False,
        checkpoint_on_cancel_enabled=False,
    )


def traced_runs(workflow: Workflow, input_data: dict) -> tuple[RunnableStatus, TracingCallbackHandler]:
    tracing = TracingCallbackHandler()
    result = workflow.run_sync(input_data=input_data, config=RunnableConfig(callbacks=[tracing]))
    return result, tracing


def run_named(tracing: TracingCallbackHandler, name: str):
    return next(run for run in tracing.runs.values() if run.name == name)


def root(tracing: TracingCallbackHandler):
    return next(run for run in tracing.runs.values() if run.type == RunType.WORKFLOW)


def flow_run(tracing: TracingCallbackHandler):
    return next(run for run in tracing.runs.values() if run.type == RunType.FLOW)


def cooling_off_workflow() -> Workflow:
    application = Input(id="application")
    score = Python(id="score", name="score", code="def run(input_data):\n    return {'score': 712}")
    score.depends_on(application)
    cooling_off = Delay(id="cooling_off", name="cooling off", duration_seconds=timedelta(days=2).total_seconds())
    cooling_off.depends_on(score)
    result = Output(id="result", name="result")
    result.depends_on(cooling_off)
    return Workflow(flow=flows.Flow(nodes=[application, score, cooling_off, result], checkpoint=runtime_checkpoint()))


class TestRunWaitingForATime:
    def test_the_trace_reads_paused_with_the_reason_and_no_error(self):
        result, tracing = traced_runs(cooling_off_workflow(), {"applicant_id": "app-1"})

        assert result.error.type is RunPausedException
        workflow_run = root(tracing)
        assert workflow_run.status == RunStatus.PAUSED
        assert workflow_run.end_time is not None
        assert workflow_run.error is None
        assert "cooling off" in workflow_run.metadata["pause"]["message"]
        assert flow_run(tracing).status == RunStatus.PAUSED
        assert flow_run(tracing).error is None

    def test_the_waiting_node_is_paused_and_finished_ones_keep_their_status(self):
        _, tracing = traced_runs(cooling_off_workflow(), {"applicant_id": "app-1"})

        delay = run_named(tracing, "cooling off")
        assert delay.status == RunStatus.PAUSED
        assert delay.error is None
        assert [execution.status for execution in delay.executions] == [RunStatus.PAUSED]
        assert run_named(tracing, "score").status == RunStatus.SUCCEEDED

    def test_the_trace_is_sent_once_the_run_pauses(self, mock_tracing_client):
        client = mock_tracing_client()
        tracing = TracingCallbackHandler(client=client)

        cooling_off_workflow().run_sync(input_data={}, config=RunnableConfig(callbacks=[tracing]))

        sent = {run.name: run.status for run in client.trace.call_args.args[0]}
        assert sent["Workflow"] == RunStatus.PAUSED
        assert sent["cooling off"] == RunStatus.PAUSED


def ask_user_workflow(answers: Queue, checkpoint: CheckpointConfig | None) -> Workflow:
    ask_user = HumanFeedbackTool(
        id="ask-user",
        name="ask-user",
        action=HumanFeedbackAction.ASK,
        input_method=FeedbackMethod.STREAM,
        output_method=FeedbackMethod.STREAM,
        streaming=StreamingConfig(enabled=True, input_queue=answers, timeout=INPUT_TIMEOUT_SECONDS),
    )
    agent = Agent(
        id="care-guide",
        name="care guide",
        llm=llms.OpenAI(
            id="care-guide-llm",
            model="gpt-4o-mini",
            connection=connections.OpenAI(api_key="test-api-key"),
            is_postponed_component_init=True,
        ),
        tools=[ask_user],
        role="Test",
        max_loops=3,
    )
    return Workflow(flow=flows.Flow(nodes=[agent], checkpoint=checkpoint or CheckpointConfig(enabled=False)))


def mock_llm_asks_the_user(mocker):
    def side_effect(stream: bool, *args, **kwargs):
        response = ModelResponse()
        response["choices"][0]["message"][
            "content"
        ] = 'Thought: Check with the user.\nAction: ask-user\nAction Input: {"input": "Which day suits you?"}'
        return response

    mocker.patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=side_effect)


class TestQuestionLeftUnansweredPastTheInputTimeout:
    def test_the_run_and_the_nodes_it_waits_on_are_paused(self, mocker):
        mock_llm_asks_the_user(mocker)

        result, tracing = traced_runs(ask_user_workflow(Queue(), runtime_checkpoint()), {"input": "Book a visit"})

        assert result.error.type is RunPausedException
        assert root(tracing).status == RunStatus.PAUSED
        assert root(tracing).error is None
        for name in ("care guide", "ask-user"):
            run = run_named(tracing, name)
            assert run.status == RunStatus.PAUSED, name
            assert run.error is None, name
            assert "timeout" in run.metadata["pause"]["message"].lower(), name
        llm_calls = [run for run in tracing.runs.values() if run.metadata.get("node", {}).get("id") == "care-guide-llm"]
        assert llm_calls
        assert {run.status for run in llm_calls} == {RunStatus.SUCCEEDED}

    def test_without_checkpointing_the_run_cannot_pause_and_fails(self, mocker):
        mock_llm_asks_the_user(mocker)

        result, tracing = traced_runs(ask_user_workflow(Queue(), None), {"input": "Book a visit"})

        assert result.status == RunnableStatus.FAILURE
        assert result.error.type is not RunPausedException
        assert root(tracing).status == RunStatus.FAILED
        assert root(tracing).error is not None
        assert run_named(tracing, "ask-user").status == RunStatus.FAILED
