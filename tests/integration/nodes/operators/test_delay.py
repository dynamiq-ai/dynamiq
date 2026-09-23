"""Delay: waits in place for up to a minute, pauses the run for longer, and passes its input on either way."""

import io
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from dynamiq import Workflow, flows
from dynamiq.checkpoints.backends.in_memory import InMemory
from dynamiq.checkpoints.config import CheckpointConfig
from dynamiq.checkpoints.types import CheckpointStatus
from dynamiq.connections.managers import get_connection_manager
from dynamiq.nodes import Behavior
from dynamiq.nodes.operators import Delay, Map
from dynamiq.nodes.utils import Input
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.serializers.loaders.yaml import WorkflowYAMLLoader
from dynamiq.types.cancellation import CancellationConfig, CancellationToken
from dynamiq.types.mocking import MockConfig


def run(delay: Delay, input_data: dict, config: RunnableConfig | None = None):
    return delay.run(input_data=input_data, config=config or RunnableConfig(callbacks=[]))


class TestWaitInPlace:
    def test_short_wait_passes_the_input_on_with_what_it_waited_for(self):
        delay = Delay(duration_seconds=0.2)

        started = time.monotonic()
        result = run(delay, {"payment_id": "pay-7", "amount": 120})
        elapsed = time.monotonic() - started

        assert result.status == RunnableStatus.SUCCESS
        assert 0.2 <= elapsed < 1.5
        assert result.output["payment_id"] == "pay-7"
        assert result.output["amount"] == 120
        assert 0.2 <= result.output["waited_seconds"] < 1.5
        assert datetime.fromisoformat(result.output["waited_until"]).tzinfo is not None

    def test_a_time_already_passed_does_not_wait(self):
        until = datetime.now(timezone.utc) - timedelta(hours=1)

        started = time.monotonic()
        result = run(Delay(), {"until": until.isoformat()})

        assert result.status == RunnableStatus.SUCCESS
        assert time.monotonic() - started < 0.5
        assert result.output["waited_until"] == until.isoformat()
        assert "until" not in result.output

    def test_a_time_without_a_zone_is_taken_as_utc(self):
        until = (datetime.now(timezone.utc) - timedelta(minutes=5)).replace(tzinfo=None, microsecond=0)

        result = run(Delay(), {"until": until.isoformat()})

        assert result.output["waited_until"] == until.replace(tzinfo=timezone.utc).isoformat()

    def test_files_pass_through_untouched(self):
        statement = io.BytesIO(b"Statement: balance 1,204.55 EUR")
        statement.name = "statement.txt"

        result = run(Delay(duration_seconds=0), {"files": [statement]})

        assert result.output["files"] == [statement]
        assert result.output["files"][0].getvalue() == b"Statement: balance 1,204.55 EUR"

    def test_the_input_duration_overrides_the_node_duration(self):
        started = time.monotonic()
        result = run(Delay(duration_seconds=3600), {"duration_seconds": 0})

        assert result.status == RunnableStatus.SUCCESS
        assert time.monotonic() - started < 0.5
        assert "duration_seconds" not in result.output

    def test_cancelling_stops_the_wait(self):
        token = CancellationToken()
        config = RunnableConfig(callbacks=[], cancellation=CancellationConfig(token=token))
        flow = flows.Flow(nodes=[Delay(id="hold", duration_seconds=30)])
        holder = {}
        worker = threading.Thread(target=lambda: holder.update(result=flow.run_sync({}, config=config)))

        started = time.monotonic()
        worker.start()
        time.sleep(0.3)
        token.cancel()
        worker.join(timeout=5)

        assert holder["result"].status == RunnableStatus.CANCELED
        assert time.monotonic() - started < 3

    @pytest.mark.asyncio
    async def test_parallel_waits_in_an_async_flow_overlap(self):
        application = Input(id="application")
        first, second = Delay(id="first", duration_seconds=0.5), Delay(id="second", duration_seconds=0.5)
        first.depends_on(application)
        second.depends_on(application)
        flow = flows.Flow(nodes=[application, first, second])

        started = time.monotonic()
        result = await flow.run_async(input_data={})

        assert result.status == RunnableStatus.SUCCESS
        assert time.monotonic() - started < 0.95


class TestInvalidWaits:
    @pytest.mark.parametrize(
        ("delay", "input_data", "message"),
        [
            (Delay(), {}, "has nothing to wait for"),
            (Delay(), {"until": "next tuesday"}, "until"),
            (Delay(), {"duration_seconds": -5}, "greater than or equal to 0"),
            (Delay(duration_seconds=1e12), {}, "too far in the future"),
        ],
        ids=["no duration", "unparseable time", "negative duration", "beyond the calendar"],
    )
    def test_is_rejected_with_a_clear_error(self, delay, input_data, message):
        result = run(delay, input_data)

        assert result.status == RunnableStatus.FAILURE
        assert message in result.error.message

    def test_a_long_wait_without_checkpointing_explains_how_to_pause_or_mock(self):
        result = run(Delay(id="cooling_off", duration_seconds=2 * 24 * 3600), {})

        assert result.status == RunnableStatus.FAILURE
        assert "needs checkpointing and the Delay at the top level" in result.error.message
        assert "mock this node" in result.error.message

    def test_a_long_wait_inside_a_map_fails_instead_of_pausing_the_run(self):
        batch = Map(id="per_invoice", node=Delay(id="hold", duration_seconds=3600), behavior=Behavior.RAISE)
        flow = flows.Flow(nodes=[batch], checkpoint=CheckpointConfig(enabled=True, backend=InMemory()))

        started = time.monotonic()
        result = flow.run_sync(input_data={"input": [{"invoice": 1}, {"invoice": 2}]})

        assert result.status == RunnableStatus.FAILURE
        assert time.monotonic() - started < 2
        assert [node.id for node in result.error.failed_nodes] == ["per_invoice"]
        assert flow._checkpoint.pending_inputs == {}
        assert flow._checkpoint.status == CheckpointStatus.FAILED

    def test_a_mocked_long_wait_returns_at_once(self):
        delay = Delay(duration_seconds=2 * 24 * 3600, mock=MockConfig(enabled=True, output={"waited_seconds": 0}))

        started = time.monotonic()
        result = run(delay, {})

        assert result.status == RunnableStatus.SUCCESS
        assert time.monotonic() - started < 0.5


def test_loads_from_the_yaml_the_platform_writes(tmp_path):
    yaml_file = tmp_path / "hold.yaml"
    yaml_file.write_text(
        """
nodes:
  application:
    type: dynamiq.nodes.utils.Input
  settlement_window:
    type: dynamiq.nodes.operators.Delay
    name: settlement window
    duration_seconds: 0.1
    depends:
      - node: application
flows:
  flow:
    nodes: [application, settlement_window]
workflows:
  wf:
    flow: flow
"""
    )

    with get_connection_manager() as cm:
        workflow = Workflow.from_yaml_file_data(
            file_data=WorkflowYAMLLoader.load(file_path=str(yaml_file), connection_manager=cm, init_components=True)
        )
    result = workflow.run(input_data={"payment_id": "pay-3"})

    delay = next(node for node in workflow.flow.nodes if node.id == "settlement_window")
    assert isinstance(delay, Delay)
    assert delay.duration_seconds == 0.1
    assert result.status == RunnableStatus.SUCCESS
    output = result.output["settlement_window"]["output"]
    assert output["application"]["output"] == {"payment_id": "pay-3"}
    assert output["waited_seconds"] >= 0.1
