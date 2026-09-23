"""A run that pauses at a waiting node and resumes from its checkpoint without repeating finished work.

The checkpoint settings are the ones the platform runtime runs with: one checkpoint per run, replaced in
place, and no save after each node, on failure or at start, so the pause itself must record the progress.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from queue import Queue
from unittest import mock

import pytest

from dynamiq import Workflow, connections, flows
from dynamiq.checkpoints.backends.filesystem import FileSystem
from dynamiq.checkpoints.backends.in_memory import InMemory
from dynamiq.checkpoints.config import CheckpointBehavior, CheckpointConfig
from dynamiq.checkpoints.types import CheckpointStatus, RunPausedException
from dynamiq.nodes import Node, NodeGroup, llms
from dynamiq.nodes.operators import Choice, ChoiceOption, Delay
from dynamiq.nodes.operators import delay as delay_module
from dynamiq.nodes.tools import Python
from dynamiq.nodes.types import ChoiceCondition, ConditionOperator
from dynamiq.nodes.utils import Input, Output
from dynamiq.prompts import Message, Prompt
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types.feedback import (
    APPROVAL_EVENT,
    ApprovalConfig,
    ApprovalInputData,
    ApprovalStreamingInputEventMessage,
    FeedbackMethod,
)
from dynamiq.types.streaming import StreamingConfig

COOLING_OFF = timedelta(days=2)
APPROVAL_TIMEOUT_SECONDS = 0.3


def runtime_checkpoint(backend) -> CheckpointConfig:
    return CheckpointConfig(
        enabled=True,
        backend=backend,
        behavior=CheckpointBehavior.REPLACE,
        checkpoint_on_start_enabled=False,
        checkpoint_after_node_enabled=False,
        checkpoint_on_failure_enabled=False,
        checkpoint_on_cancel_enabled=False,
    )


def library_default_checkpoint(backend) -> CheckpointConfig:
    """Checkpointing as a library user turns it on: a new snapshot at the start and after every node."""
    return CheckpointConfig(enabled=True, backend=backend)


@pytest.fixture(params=[runtime_checkpoint, library_default_checkpoint], ids=["runtime", "library-defaults"])
def checkpointing(request):
    return request.param


@pytest.fixture(params=["in_memory", "filesystem"])
def backend(request, tmp_path):
    if request.param == "filesystem":
        return FileSystem(base_path=str(tmp_path / "checkpoints"))
    return InMemory()


class Ledger:
    """Counts how often each step really ran, the property a resume must not break."""

    def __init__(self):
        self.calls: dict[str, int] = {}

    def step(self, node_id: str, code: str, **kwargs) -> Python:
        ledger = self

        class CountedPython(Python):
            def execute(self, input_data, config=None, **run_kwargs):
                ledger.calls[self.id] = ledger.calls.get(self.id, 0) + 1
                return super().execute(input_data, config, **run_kwargs)

        return CountedPython(id=node_id, name=node_id, code=code, **kwargs)


def at(moment: datetime):
    """Moves the Delay's clock, as if the resume came at `moment`."""

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return moment

    return mock.patch.object(delay_module, "datetime", Clock)


def loan_flow(ledger: Ledger, backend, checkpointing=runtime_checkpoint) -> flows.Flow:
    """A loan disbursement held for a statutory cooling-off period after scoring."""
    application = Input(id="application")
    score = ledger.step("credit_score", "def run(input_data):\n    return {'score': 712, 'band': 'B'}")
    score.depends_on(application)
    cooling_off = Delay(id="cooling_off", name="cooling off", duration_seconds=COOLING_OFF.total_seconds())
    cooling_off.depends_on(score)
    disburse = ledger.step("disburse", "def run(input_data):\n    return {'disbursed': True}")
    disburse.depends_on(cooling_off)
    result = Output(id="result")
    result.depends_on(disburse)
    return flows.Flow(
        id="loan-disbursement",
        nodes=[application, score, cooling_off, disburse, result],
        checkpoint=checkpointing(backend),
    )


class TestLoanCoolingOff:
    def test_pause_records_finished_steps_and_the_resume_time(self, backend, checkpointing):
        ledger = Ledger()
        flow = loan_flow(ledger, backend, checkpointing)
        started = datetime.now(timezone.utc)

        result = flow.run_sync(input_data={"applicant_id": "app-1", "amount": 25_000})

        assert result.status == RunnableStatus.FAILURE
        assert result.error.type is RunPausedException
        assert "cooling off" in result.error.message
        saved = backend.load(flow._checkpoint.id)
        assert saved.status == CheckpointStatus.PENDING_INPUT
        assert set(saved.completed_node_ids) == {"application", "credit_score"}
        assert list(saved.pending_inputs) == ["cooling_off"]
        assert started + COOLING_OFF <= saved.resume_at <= datetime.now(timezone.utc) + COOLING_OFF
        assert saved.get_node_output("credit_score") == {"content": {"score": 712, "band": "B"}}
        assert ledger.calls == {"credit_score": 1}

    def test_workflow_reports_the_pause_as_a_pause(self, backend, checkpointing, caplog):
        workflow = Workflow(flow=loan_flow(Ledger(), backend, checkpointing))

        result = workflow.run_sync(input_data={"applicant_id": "app-1", "amount": 25_000})

        assert result.status == RunnableStatus.FAILURE
        assert result.error.type is RunPausedException
        assert "execution paused" in caplog.text
        assert "execution failed" not in caplog.text

    def test_early_resume_pauses_again_for_the_same_time(self, backend, checkpointing):
        ledger = Ledger()
        flow = loan_flow(ledger, backend, checkpointing)
        flow.run_sync(input_data={"applicant_id": "app-1", "amount": 25_000})
        checkpoint_id, resume_at = flow._checkpoint.id, flow._checkpoint.resume_at

        with at(resume_at - timedelta(hours=1)):
            result = flow.run_sync(input_data=None, resume_from=checkpoint_id)

        assert result.error.type is RunPausedException
        assert backend.load(checkpoint_id).resume_at == resume_at
        assert ledger.calls == {"credit_score": 1}

    def test_resume_on_time_finishes_with_every_step_run_once(self, backend, checkpointing):
        ledger = Ledger()
        flow = loan_flow(ledger, backend, checkpointing)
        flow.run_sync(input_data={"applicant_id": "app-1", "amount": 25_000})
        checkpoint_id, resume_at = flow._checkpoint.id, flow._checkpoint.resume_at

        with at(resume_at + timedelta(seconds=1)):
            result = flow.run_sync(input_data=None, resume_from=checkpoint_id)

        assert result.status == RunnableStatus.SUCCESS
        assert ledger.calls == {"credit_score": 1, "disburse": 1}
        waited = result.output["cooling_off"]["output"]
        assert waited["waited_until"] == resume_at.isoformat()
        assert waited["waited_seconds"] >= COOLING_OFF.total_seconds()
        assert result.output["result"]["status"] == RunnableStatus.SUCCESS.value

    @pytest.mark.asyncio
    async def test_async_run_pauses_and_resumes_like_a_sync_one(self, backend, checkpointing):
        ledger = Ledger()
        flow = loan_flow(ledger, backend, checkpointing)

        paused = await flow.run_async(input_data={"applicant_id": "app-1", "amount": 25_000})
        checkpoint_id, resume_at = flow._checkpoint.id, flow._checkpoint.resume_at
        with at(resume_at + timedelta(seconds=1)):
            resumed = await flow.run_async(input_data=None, resume_from=checkpoint_id)

        assert paused.error.type is RunPausedException
        assert resumed.status == RunnableStatus.SUCCESS
        assert ledger.calls == {"credit_score": 1, "disburse": 1}


class TestOneFlowManyRuns:
    def test_each_run_waits_its_own_time_on_a_flow_that_resumed_another(self):
        """A service keeps one Flow and resumes many runs through it: what one run restored into a node must not
        cut the next run's wait short."""
        application = Input(id="application")
        cooling_off = Delay(id="cooling_off", duration_seconds=COOLING_OFF.total_seconds())
        cooling_off.depends_on(application)
        settlement = Delay(id="settlement", duration_seconds=timedelta(days=1).total_seconds())
        settlement.depends_on(cooling_off)
        flow = flows.Flow(nodes=[application, cooling_off, settlement], checkpoint=runtime_checkpoint(InMemory()))
        started = datetime.now(timezone.utc)
        with at(started):
            flow.run_sync(input_data={"loan": "B"})
            loan_b = flow._checkpoint.id
            flow.run_sync(input_data={"loan": "A"})
        with at(started + COOLING_OFF + timedelta(minutes=1)):
            flow.run_sync(input_data=None, resume_from=flow._checkpoint.id)
        later = started + COOLING_OFF + timedelta(days=1, minutes=2)
        with at(later):
            loan_a = flow.run_sync(input_data=None, resume_from=flow._checkpoint.id)
            loan_b_resumed = flow.run_sync(input_data=None, resume_from=loan_b)

        assert loan_a.status == RunnableStatus.SUCCESS
        assert loan_b_resumed.error.type is RunPausedException
        assert flow._checkpoint.resume_at == later + timedelta(days=1)


class TestPauseKeepsTheRestOfTheFlowMoving:
    def test_independent_branch_finishes_before_the_pause_and_is_not_repeated(self):
        ledger = Ledger()
        backend = InMemory()
        application = Input(id="application")
        kyc = ledger.step("kyc_check", "def run(input_data):\n    return {'kyc': 'passed'}")
        kyc.depends_on(application)
        settlement_window = Delay(id="settlement_window", duration_seconds=3600)
        settlement_window.depends_on(application)
        release_funds = ledger.step("release_funds", "def run(input_data):\n    return {'released': True}")
        release_funds.depends_on([kyc, settlement_window])
        flow = flows.Flow(
            nodes=[application, kyc, settlement_window, release_funds], checkpoint=runtime_checkpoint(backend)
        )

        paused = flow.run_sync(input_data={"payment_id": "pay-9"})

        assert paused.error.type is RunPausedException
        assert paused.output["kyc_check"]["status"] == RunnableStatus.SUCCESS.value
        assert paused.output["release_funds"]["status"] == RunnableStatus.UNDEFINED.value
        checkpoint_id, resume_at = flow._checkpoint.id, flow._checkpoint.resume_at
        with at(resume_at + timedelta(seconds=1)):
            resumed = flow.run_sync(input_data=None, resume_from=checkpoint_id)
        assert resumed.status == RunnableStatus.SUCCESS
        assert ledger.calls == {"kyc_check": 1, "release_funds": 1}

    def test_nodes_behind_a_waiting_node_that_returns_errors_do_not_run(self):
        ledger = Ledger()
        application = Input(id="application")
        hold = Delay(id="hold", duration_seconds=3600, error_handling={"behavior": "return"})
        hold.depends_on(application)
        notify = ledger.step("notify_customer", "def run(input_data):\n    return {'sent': True}")
        notify.depends_on(hold)
        flow = flows.Flow(nodes=[application, hold, notify], checkpoint=runtime_checkpoint(InMemory()))

        paused = flow.run_sync(input_data={})

        assert paused.error.type is RunPausedException
        assert ledger.calls == {}
        assert paused.output["notify_customer"]["status"] == RunnableStatus.UNDEFINED.value

    def test_a_real_failure_elsewhere_fails_the_run_instead_of_pausing_it(self):
        application = Input(id="application")
        sanctions = Python(id="sanctions_screening", code="def run(input_data):\n    raise ValueError('list down')")
        sanctions.depends_on(application)
        hold = Delay(id="hold", duration_seconds=3600)
        hold.depends_on(application)
        flow = flows.Flow(nodes=[application, sanctions, hold], checkpoint=runtime_checkpoint(InMemory()))

        result = flow.run_sync(input_data={})

        assert result.status == RunnableStatus.FAILURE
        assert [node.id for node in result.error.failed_nodes] == ["sanctions_screening"]
        assert flow._checkpoint.status == CheckpointStatus.FAILED

    def test_a_pause_that_cannot_be_saved_fails_the_run(self):
        backend = InMemory()
        flow = loan_flow(Ledger(), backend)

        with mock.patch.object(InMemory, "save", side_effect=ConnectionError("checkpoint store unreachable")):
            result = flow.run_sync(input_data={"applicant_id": "app-1"})

        assert result.status == RunnableStatus.FAILURE
        assert result.error.type is ConnectionError
        assert flow._checkpoint.status == CheckpointStatus.FAILED


class Statement(Node):
    """Rows as a SQL query returns them: tuples with Decimal amounts, and totals keyed by year."""

    group: NodeGroup = NodeGroup.UTILS
    name: str = "statement"

    def execute(self, input_data, config=None, **kwargs):
        return {
            "rows": [("ACC-1", Decimal("1520.35")), ("ACC-2", Decimal("-40.10"))],
            "totals_by_year": {2025: Decimal("12000.00"), 2026: Decimal("9150.75")},
        }


class Received(Node):
    """Keeps what it was given, so a test can compare a resumed run with an uninterrupted one."""

    group: NodeGroup = NodeGroup.UTILS
    name: str = "received"

    def execute(self, input_data, config=None, **kwargs):
        return {"statement": input_data["statement"]["output"]}


class TestPauseKeepsValuesExact:
    def test_amounts_and_rows_come_back_exactly_after_a_pause(self, backend):
        application = Input(id="application")
        statement = Statement(id="statement")
        statement.depends_on(application)
        review_period = Delay(id="review_period", duration_seconds=COOLING_OFF.total_seconds())
        review_period.depends_on(statement)
        received = Received(id="received")
        received.depends_on([statement, review_period])
        flow = flows.Flow(
            nodes=[application, statement, review_period, received], checkpoint=runtime_checkpoint(backend)
        )

        flow.run_sync(input_data={})
        with at(flow._checkpoint.resume_at + timedelta(seconds=1)):
            resumed = flow.run_sync(input_data=None, resume_from=flow._checkpoint.id)

        assert resumed.status == RunnableStatus.SUCCESS
        got = resumed.output["received"]["output"]["statement"]
        assert got == Statement(id="statement").execute({})
        assert isinstance(got["rows"][0], tuple)
        assert isinstance(got["totals_by_year"][2025], Decimal)


class TestChoiceBeforeAPause:
    def test_branch_chosen_before_the_pause_is_honoured_after_it(self):
        ledger = Ledger()
        application = Input(id="application")
        routing = Choice(
            id="routing",
            options=[
                ChoiceOption(
                    id="manual_review",
                    condition=ChoiceCondition(
                        variable="$.amount", operator=ConditionOperator.NUMERIC_GREATER_THAN, value=10_000
                    ),
                ),
                ChoiceOption(id="straight_through"),
            ],
        )
        routing.depends_on(application)
        review_hold = Delay(id="review_hold", duration_seconds=24 * 3600)
        review_hold.depends.append(routing.depends[0].model_copy(update={"node": routing, "option": "manual_review"}))
        underwriter = ledger.step("underwriter", "def run(input_data):\n    return {'reviewed': True}")
        underwriter.depends_on(review_hold)
        flow = flows.Flow(
            nodes=[application, routing, review_hold, underwriter], checkpoint=runtime_checkpoint(InMemory())
        )

        flow.run_sync(input_data={"amount": 50_000})
        checkpoint_id, resume_at = flow._checkpoint.id, flow._checkpoint.resume_at
        with at(resume_at + timedelta(seconds=1)):
            resumed = flow.run_sync(input_data=None, resume_from=checkpoint_id)

        assert resumed.status == RunnableStatus.SUCCESS
        assert ledger.calls == {"underwriter": 1}


class TestHumanApprovalResume:
    def test_steps_before_an_approval_are_not_repeated_after_it(self, mock_llm_executor, checkpointing):
        queue = Queue()
        application = Input(id="application")
        summary = llms.OpenAI(
            id="risk_summary",
            model="gpt-4o-mini",
            connection=connections.OpenAI(api_key="test-api-key"),
            prompt=Prompt(messages=[Message(role="user", content="Summarise the risk of {{application}}")]),
            is_postponed_component_init=True,
        )
        summary.depends_on(application)
        payout = Python(
            id="payout",
            code="def run(input_data):\n    return {'paid': True}",
            approval=ApprovalConfig(enabled=True, feedback_method=FeedbackMethod.STREAM),
            streaming=StreamingConfig(enabled=True, input_queue=queue, timeout=APPROVAL_TIMEOUT_SECONDS),
        )
        payout.depends_on(summary)
        flow = flows.Flow(nodes=[application, summary, payout], checkpoint=checkpointing(InMemory()))

        paused = flow.run_sync(input_data={"application": "claim 42"})
        queue.put(
            ApprovalStreamingInputEventMessage(
                entity_id="payout", event=APPROVAL_EVENT, data=ApprovalInputData(is_approved=True, feedback="")
            ).model_dump_json()
        )
        resumed = flow.run_sync(input_data=None, resume_from=flow._checkpoint.id, config=RunnableConfig())

        assert paused.status == RunnableStatus.FAILURE
        assert flow._checkpoint.pending_inputs == {}
        assert resumed.status == RunnableStatus.SUCCESS
        assert resumed.output["payout"]["output"] == {"content": {"paid": True}}
        assert mock_llm_executor.call_count == 1
