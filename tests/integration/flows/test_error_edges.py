"""Error edges: a dependency with `trigger: failure` runs its node only when the upstream node failed.

The scenario is a card payment with compensation, the saga pattern a payments team builds: charge the card,
and when the charge fails for good, refund or notify instead of failing the whole run.
"""

import os
from datetime import datetime, timedelta
from queue import Queue
from unittest import mock

import pytest
from pydantic import ValidationError

from dynamiq import Workflow, flows
from dynamiq.checkpoints.backends.in_memory import InMemory
from dynamiq.checkpoints.config import CheckpointBehavior, CheckpointConfig
from dynamiq.checkpoints.types import RunPausedException
from dynamiq.connections.managers import get_connection_manager
from dynamiq.nodes import DependencyTrigger, ErrorHandling
from dynamiq.nodes.node import NodeDependency
from dynamiq.nodes.operators import Choice, ChoiceOption, Delay
from dynamiq.nodes.operators import delay as delay_module
from dynamiq.nodes.tools import Python
from dynamiq.nodes.types import ChoiceCondition, ConditionOperator
from dynamiq.nodes.utils import Input, Output
from dynamiq.runnables import RunnableStatus
from dynamiq.serializers.loaders.yaml import WorkflowYAMLLoader, WorkflowYAMLLoaderException
from dynamiq.types.feedback import (
    APPROVAL_EVENT,
    ApprovalConfig,
    ApprovalInputData,
    ApprovalStreamingInputEventMessage,
    FeedbackMethod,
)
from dynamiq.types.streaming import StreamingConfig

DECLINED = "def run(input_data):\n    raise RuntimeError('card declined: insufficient funds')"
GATEWAY_TIMEOUT = "def run(input_data):\n    raise RuntimeError('gateway timeout')"
CHARGED = "def run(input_data):\n    return {'charged': 120}"
REFUND = "def run(input_data):\n    return {'refunded': True, 'reason': input_data['charge_card']['error']['message']}"
RECEIPT = "def run(input_data):\n    return {'receipt': 'r-1'}"


class Calls:
    """Counts how often each step ran, retries included."""

    def __init__(self):
        self.count: dict[str, int] = {}

    def step(self, node_id: str, code: str, **kwargs) -> Python:
        calls = self

        class CountedPython(Python):
            def execute(self, input_data, config=None, **run_kwargs):
                calls.count[self.id] = calls.count.get(self.id, 0) + 1
                return super().execute(input_data, config, **run_kwargs)

        return CountedPython(id=node_id, name=node_id, code=code, **kwargs)


def payment_flow(calls: Calls, charge_code: str, refund_code: str = REFUND) -> flows.Flow:
    """Charge the card, then send a receipt; when the charge fails for good, refund instead."""
    order = Input(id="order")
    charge = calls.step(
        "charge_card", charge_code, error_handling=ErrorHandling(max_retries=2, retry_interval_seconds=0.01)
    )
    charge.depends_on(order)
    receipt = calls.step("send_receipt", RECEIPT)
    receipt.depends_on(charge)
    refund = calls.step("refund", refund_code)
    refund.depends_on(charge, trigger=DependencyTrigger.FAILURE)
    result = Output(id="result")
    result.depends_on([receipt, refund])
    return flows.Flow(id="card-payment", nodes=[order, charge, receipt, refund, result])


def statuses(result) -> dict[str, str]:
    return {node_id: node_result["status"] for node_id, node_result in result.output.items()}


def durable() -> CheckpointConfig:
    """Checkpointing as the platform runs flows: one checkpoint per run, saved when the run pauses."""
    return CheckpointConfig(
        enabled=True, backend=InMemory(), behavior=CheckpointBehavior.REPLACE, checkpoint_after_node_enabled=False
    )


def after(resume_at: datetime):
    """Moves the Delay's clock past `resume_at`, as if the run resumed on time."""

    class AfterTheHold(datetime):
        @classmethod
        def now(cls, tz=None):
            return resume_at + timedelta(seconds=1)

    return mock.patch.object(delay_module, "datetime", AfterTheHold)


class TestCompensation:
    def test_refund_runs_once_retries_are_exhausted_and_the_run_succeeds(self):
        calls = Calls()

        result = payment_flow(calls, DECLINED).run_sync(input_data={"order_id": "o-1", "amount": 120})

        assert result.status == RunnableStatus.SUCCESS
        assert calls.count == {"charge_card": 3, "refund": 1}
        assert statuses(result) == {
            "order": "success",
            "charge_card": "failure",
            "send_receipt": "skip",
            "refund": "success",
            "result": "success",
        }
        assert result.output["refund"]["output"]["content"] == {
            "refunded": True,
            "reason": "Code execution error: card declined: insufficient funds",
        }

    def test_refund_is_skipped_when_the_charge_succeeds(self):
        calls = Calls()

        result = payment_flow(calls, CHARGED).run_sync(input_data={"order_id": "o-2", "amount": 120})

        assert result.status == RunnableStatus.SUCCESS
        assert calls.count == {"charge_card": 1, "send_receipt": 1}
        assert statuses(result)["refund"] == "skip"
        assert statuses(result)["result"] == "success"

    def test_a_failing_refund_fails_the_run_and_names_only_the_refund(self):
        calls = Calls()
        refund_down = "def run(input_data):\n    raise RuntimeError('refund service unavailable')"

        result = payment_flow(calls, DECLINED, refund_down).run_sync(input_data={"order_id": "o-3", "amount": 120})

        assert result.status == RunnableStatus.FAILURE
        assert [node.id for node in result.error.failed_nodes] == ["refund"]
        assert "refund service unavailable" in result.error.failed_nodes[0].error_message

    @pytest.mark.asyncio
    async def test_async_run_routes_the_failure_like_a_sync_one(self):
        calls = Calls()

        result = await payment_flow(calls, DECLINED).run_async(input_data={"order_id": "o-4", "amount": 120})

        assert result.status == RunnableStatus.SUCCESS
        assert calls.count == {"charge_card": 3, "refund": 1}
        assert statuses(result)["result"] == "success"


def plain_step(node_id: str, code: str) -> Python:
    return Python(id=node_id, name=node_id, code=code)


class TestRoutingByError:
    @staticmethod
    def flow_with_handlers(step, charge_code: str) -> flows.Flow:
        order = Input(id="order")
        charge = step("charge_card", charge_code)
        charge.depends_on(order)
        notify = step("notify_customer", "def run(input_data):\n    return {'sent': True}")
        notify.depends_on(
            charge,
            trigger=DependencyTrigger.FAILURE,
            condition=ChoiceCondition(
                variable="$.error.message", operator=ConditionOperator.STRING_CONTAINS, value="declined"
            ),
        )
        retry_later = step("schedule_retry", "def run(input_data):\n    return {'queued': True}")
        retry_later.depends_on(
            charge,
            trigger=DependencyTrigger.FAILURE,
            condition=ChoiceCondition(
                variable="$.error.message", operator=ConditionOperator.STRING_CONTAINS, value="timeout"
            ),
        )
        return flows.Flow(nodes=[order, charge, notify, retry_later])

    def test_each_error_reaches_only_its_own_handler(self):
        declined, timed_out = Calls(), Calls()

        declined_result = self.flow_with_handlers(declined.step, DECLINED).run_sync(input_data={})
        timed_out_result = self.flow_with_handlers(timed_out.step, GATEWAY_TIMEOUT).run_sync(input_data={})

        assert declined_result.status == RunnableStatus.SUCCESS
        assert declined.count == {"charge_card": 1, "notify_customer": 1}
        assert timed_out_result.status == RunnableStatus.SUCCESS
        assert timed_out.count == {"charge_card": 1, "schedule_retry": 1}

    def test_an_error_no_handler_matches_fails_the_run(self):
        calls = Calls()
        fraud = "def run(input_data):\n    raise RuntimeError('fraud suspected')"

        result = self.flow_with_handlers(calls.step, fraud).run_sync(input_data={})

        assert result.status == RunnableStatus.FAILURE
        assert [node.id for node in result.error.failed_nodes] == ["charge_card"]
        assert calls.count == {"charge_card": 1}


class TestErrorEdgesAndOtherBranching:
    def test_output_joins_a_longer_success_branch_with_the_handler(self):
        calls = Calls()
        order = Input(id="order")
        charge = calls.step("charge_card", DECLINED)
        charge.depends_on(order)
        book = calls.step("book_ledger", "def run(input_data):\n    return {'booked': True}")
        book.depends_on(charge)
        receipt = calls.step("send_receipt", RECEIPT)
        receipt.depends_on(book)
        refund = calls.step("refund", REFUND)
        refund.depends_on(charge, trigger=DependencyTrigger.FAILURE)
        result = Output(id="result")
        result.depends_on([order, receipt, refund])

        run = flows.Flow(nodes=[order, charge, book, receipt, refund, result]).run_sync(input_data={})

        assert run.status == RunnableStatus.SUCCESS
        assert statuses(run)["result"] == "success"
        assert calls.count == {"charge_card": 1, "refund": 1}

    def test_a_rejected_approval_is_not_a_failure_and_does_not_fire_the_error_edge(self):
        calls = Calls()
        queue = Queue()
        queue.put(
            ApprovalStreamingInputEventMessage(
                entity_id="charge_card",
                event=APPROVAL_EVENT,
                data=ApprovalInputData(is_approved=False, feedback="amount looks wrong"),
            ).model_dump_json()
        )
        order = Input(id="order")
        charge = calls.step(
            "charge_card",
            CHARGED,
            approval=ApprovalConfig(enabled=True, feedback_method=FeedbackMethod.STREAM),
            streaming=StreamingConfig(enabled=True, input_queue=queue, timeout=5),
        )
        charge.depends_on(order)
        refund = calls.step("refund", REFUND)
        refund.depends_on(charge, trigger=DependencyTrigger.FAILURE)

        result = flows.Flow(nodes=[order, charge, refund]).run_sync(input_data={})

        assert result.status == RunnableStatus.SUCCESS
        assert statuses(result) == {"order": "success", "charge_card": "skip", "refund": "skip"}
        assert calls.count == {}

    def test_a_pause_is_not_a_failure_and_does_not_fire_the_error_edge(self):
        calls = Calls()
        order = Input(id="order")
        hold = Delay(id="fraud_hold", duration_seconds=3600)
        hold.depends_on(order)
        escalate = calls.step("escalate", "def run(input_data):\n    return {'escalated': True}")
        escalate.depends_on(hold, trigger=DependencyTrigger.FAILURE)
        release = calls.step("release", "def run(input_data):\n    return {'released': True}")
        release.depends_on(hold)
        flow = flows.Flow(nodes=[order, hold, escalate, release], checkpoint=durable())

        paused = flow.run_sync(input_data={})
        with after(flow._checkpoint.resume_at):
            resumed = flow.run_sync(input_data=None, resume_from=flow._checkpoint.id)

        assert paused.error.type is RunPausedException
        assert statuses(paused)["escalate"] == "undefined"
        assert resumed.status == RunnableStatus.SUCCESS
        assert statuses(resumed)["escalate"] == "skip"
        assert calls.count == {"release": 1}

    def test_a_handler_that_waits_still_matches_the_error_type_when_the_run_resumes(self):
        """Retry a declined charge the next day: the handler's condition is checked again on resume, against the
        failure the checkpoint restored."""
        calls = Calls()
        order = Input(id="order")
        charge = calls.step("charge_card", DECLINED)
        charge.depends_on(order)
        wait_a_day = Delay(id="wait_a_day", duration_seconds=timedelta(days=1).total_seconds())
        wait_a_day.depends_on(
            charge,
            trigger=DependencyTrigger.FAILURE,
            condition=ChoiceCondition(
                variable="$.error.type", operator=ConditionOperator.STRING_EQUALS, value="ToolExecutionException"
            ),
        )
        retry = calls.step("retry_charge", CHARGED)
        retry.depends_on(wait_a_day)
        flow = flows.Flow(nodes=[order, charge, wait_a_day, retry], checkpoint=durable())

        paused = flow.run_sync(input_data={"order_id": "o-9"})
        with after(flow._checkpoint.resume_at):
            resumed = flow.run_sync(input_data=None, resume_from=flow._checkpoint.id)

        assert paused.error.type is RunPausedException
        assert resumed.status == RunnableStatus.SUCCESS
        assert statuses(resumed) == {
            "order": "success",
            "charge_card": "failure",
            "wait_a_day": "success",
            "retry_charge": "success",
        }
        assert calls.count == {"charge_card": 1, "retry_charge": 1}

    @pytest.mark.parametrize(
        "charge_code, receipt", [(CHARGED, "success"), (DECLINED, "skip")], ids=["paid", "declined"]
    )
    def test_an_output_on_the_error_edge_reports_either_outcome(self, charge_code, receipt):
        calls = Calls()
        order = Input(id="order")
        charge = calls.step("charge_card", charge_code)
        charge.depends_on(order)
        send_receipt = calls.step("send_receipt", RECEIPT)
        send_receipt.depends_on(charge)
        result = Output(id="result")
        result.depends_on(send_receipt)
        result.depends_on(charge, trigger=DependencyTrigger.FAILURE)

        run = flows.Flow(nodes=[order, charge, send_receipt, result]).run_sync(input_data={})

        assert run.status == RunnableStatus.SUCCESS
        assert statuses(run)["send_receipt"] == receipt
        assert statuses(run)["result"] == "success"

    def test_a_handler_that_also_waits_on_a_pause_takes_the_failure_over_after_the_resume(self):
        """Refund a declined charge once the settlement window closes: the run pauses with the failure handled
        later, not failed now, and the refund runs on resume."""
        calls = Calls()
        order = Input(id="order")
        charge = calls.step("charge_card", DECLINED)
        charge.depends_on(order)
        settlement_window = Delay(id="settlement_window", duration_seconds=3600)
        settlement_window.depends_on(order)
        refund = calls.step("refund", REFUND)
        refund.depends_on(charge, trigger=DependencyTrigger.FAILURE)
        refund.depends_on(settlement_window)
        flow = flows.Flow(nodes=[order, charge, settlement_window, refund], checkpoint=durable())

        paused = flow.run_sync(input_data={})
        with after(flow._checkpoint.resume_at):
            resumed = flow.run_sync(input_data=None, resume_from=flow._checkpoint.id)

        assert paused.error.type is RunPausedException
        assert resumed.status == RunnableStatus.SUCCESS
        assert statuses(resumed)["refund"] == "success"
        assert calls.count == {"charge_card": 1, "refund": 1}


class TestDefinition:
    def test_failure_trigger_cannot_select_a_choice_option(self):
        routing = Choice(id="routing", options=[ChoiceOption(id="manual")])

        with pytest.raises(ValidationError, match="cannot both select option 'manual' and trigger on failure"):
            NodeDependency(node=routing, option="manual", trigger=DependencyTrigger.FAILURE)

    def test_yaml_round_trip_keeps_triggers_and_conditions(self, tmp_path):
        flow = TestRoutingByError.flow_with_handlers(plain_step, DECLINED)
        yaml_file = os.path.join(tmp_path, "payment.yaml")
        Workflow(id="payment", flow=flow).to_yaml_file(yaml_file)

        with get_connection_manager() as cm:
            loaded = Workflow.from_yaml_file_data(
                file_data=WorkflowYAMLLoader.load(file_path=yaml_file, connection_manager=cm, init_components=True)
            )
        loaded_nodes = {node.id: node for node in loaded.flow.nodes}
        [notify_dependency] = loaded_nodes["notify_customer"].depends
        [charge_dependency] = loaded_nodes["charge_card"].depends
        result = loaded.flow.run_sync(input_data={})

        assert notify_dependency.trigger == DependencyTrigger.FAILURE
        assert notify_dependency.condition.variable == "$.error.message"
        assert notify_dependency.condition.value == "declined"
        assert charge_dependency.trigger == DependencyTrigger.SUCCESS
        assert "trigger" not in open(yaml_file).read().split("charge_card:")[1].split("notify_customer:")[0]
        assert result.status == RunnableStatus.SUCCESS
        assert statuses(result)["notify_customer"] == "success"
        assert statuses(result)["schedule_retry"] == "skip"

    def test_yaml_with_a_failure_trigger_on_an_option_is_rejected(self, tmp_path):
        yaml_file = tmp_path / "invalid.yaml"
        yaml_file.write_text(
            """
nodes:
  routing:
    type: dynamiq.nodes.operators.Choice
    options:
      - id: manual
  review:
    type: dynamiq.nodes.utils.Output
    depends:
      - node: routing
        option: manual
        trigger: failure
flows:
  flow:
    nodes: [routing, review]
workflows:
  wf:
    flow: flow
"""
        )

        with get_connection_manager() as cm, pytest.raises(WorkflowYAMLLoaderException, match="trigger on failure"):
            WorkflowYAMLLoader.load(file_path=str(yaml_file), connection_manager=cm, init_components=True)
