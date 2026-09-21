import copy
import time
from typing import ClassVar

import pytest
from pydantic import PrivateAttr

from dynamiq.flows import Flow
from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.dry_run import DryRunMixin
from dynamiq.nodes.node import ErrorHandling, NodeDependency
from dynamiq.nodes.operators import Map, Pass, SubWorkflow
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types.dry_run import DryRunConfig


class RecordingStore(DryRunMixin):
    """Stands in for a vector store: records each delete it is asked for."""

    def __init__(self):
        super().__init__()
        self.deletes: list[list[str]] = []

    def delete_documents(self, document_ids: list[str] | None = None, delete_all: bool = False) -> None:
        self.deletes.append(list(document_ids or []))


class Ingest(Node):
    """Stands in for a writer: records the cleanup of what it wrote; counted on the class since a flow runs copies."""

    group: NodeGroup = NodeGroup.UTILS
    name: str = "ingest"
    cleanups: ClassVar[list[str]] = []
    _written: bool = PrivateAttr(default=False)

    def execute(self, input_data, config=None, **kwargs):
        self._written = True
        return {"written": True}

    def dry_run_cleanup(self, dry_run_config: DryRunConfig | None = None) -> None:
        if self._written:
            self.cleanups.append(self.id)


class Slow(Node):
    group: NodeGroup = NodeGroup.UTILS
    name: str = "slow"

    def execute(self, input_data, config=None, **kwargs):
        time.sleep(0.3)
        return {"done": True}


@pytest.fixture(autouse=True)
def reset_cleanups():
    Ingest.cleanups.clear()


def dry_run_config() -> RunnableConfig:
    return RunnableConfig(dry_run=DryRunConfig(enabled=True), callbacks=[])


def test_a_store_copied_with_its_node_cleans_the_shared_tracked_documents_once():
    store = RecordingStore()
    twin = copy.copy(store)
    twin._track_documents(["a", "b"])

    store.dry_run_cleanup(DryRunConfig(enabled=True))
    twin.dry_run_cleanup(DryRunConfig(enabled=True))

    assert store.deletes == [["a", "b"]]
    assert store._tracked_documents == [] and twin._tracked_documents == []


def test_a_flow_run_cleans_up_at_its_end_by_default():
    result = Flow(id="flow", nodes=[Ingest(id="writer")]).run_sync({}, dry_run_config())

    assert result.status == RunnableStatus.SUCCESS
    assert Ingest.cleanups == ["writer"]


def test_a_flow_run_told_not_to_clean_up_leaves_it_to_the_hook():
    flow = Flow(id="flow", nodes=[Ingest(id="writer")])
    config = dry_run_config()

    result = flow.run_sync({}, config, cleanup_dry_run=False)

    assert result.status == RunnableStatus.SUCCESS
    assert Ingest.cleanups == []
    flow.dry_run_cleanup(config.dry_run)
    assert Ingest.cleanups == ["writer"]


@pytest.mark.asyncio
async def test_an_async_flow_run_told_not_to_clean_up_leaves_it_to_the_hook():
    flow = Flow(id="flow", nodes=[Ingest(id="writer")])
    config = dry_run_config()

    result = await flow.run_async({}, config, cleanup_dry_run=False)

    assert result.status == RunnableStatus.SUCCESS
    assert Ingest.cleanups == []
    flow.dry_run_cleanup(config.dry_run)
    assert Ingest.cleanups == ["writer"]


def test_a_sub_workflow_keeps_the_copy_that_ran_for_the_hook_and_a_clone_starts_empty():
    node = SubWorkflow(id="ingest", flow=Flow(id="inner", nodes=[Ingest(id="writer")]))
    config = dry_run_config()

    assert node.run({}, config).status == RunnableStatus.SUCCESS
    assert Ingest.cleanups == []
    assert len(node._dry_run_flows) == 1
    assert node.clone()._dry_run_flows == []

    node.dry_run_cleanup(config.dry_run)

    assert Ingest.cleanups == ["writer"]
    assert node._dry_run_flows == []


def test_a_sub_workflow_keeps_nothing_outside_a_dry_run():
    node = SubWorkflow(id="ingest", flow=Flow(id="inner", nodes=[Ingest(id="writer")]))

    assert node.run({}, RunnableConfig(callbacks=[])).status == RunnableStatus.SUCCESS
    assert node._dry_run_flows == []


def test_a_map_cleans_the_clones_that_ran_per_item_and_a_clone_starts_empty():
    batch = Map(id="batch", node=Ingest(id="writer"))
    config = dry_run_config()

    result = batch.run({"input": [{"n": 1}, {"n": 2}]}, config)

    assert result.status == RunnableStatus.SUCCESS
    assert result.output["output"] == [{"written": True}, {"written": True}]
    assert Ingest.cleanups == []
    assert len(batch._dry_run_nodes) == 2
    assert batch.clone()._dry_run_nodes == []

    batch.dry_run_cleanup(config.dry_run)

    # One clone per item, each under the id the item ran with; the template never ran.
    assert len(Ingest.cleanups) == 2 and len(set(Ingest.cleanups)) == 2 and "writer" not in Ingest.cleanups
    assert batch._dry_run_nodes == []


def test_a_map_still_cleans_the_clones_when_the_template_cannot_be_cleaned():
    class Unconnected(Ingest):
        def dry_run_cleanup(self, dry_run_config: DryRunConfig | None = None) -> None:
            if not self._written:
                raise AttributeError("'NoneType' object has no attribute 'dry_run_cleanup'")
            super().dry_run_cleanup(dry_run_config)

    batch = Map(id="batch", node=Unconnected(id="writer"))
    config = dry_run_config()
    assert batch.run({"input": [{"n": 1}, {"n": 2}]}, config).status == RunnableStatus.SUCCESS

    batch.dry_run_cleanup(config.dry_run)

    assert len(Ingest.cleanups) == 2


def test_a_copy_that_outlives_the_holder_cleans_up_when_it_ends():
    slow = Slow(id="slow")
    writer = Ingest(id="writer", depends=[NodeDependency(node=slow)])
    node = SubWorkflow(
        id="ingest",
        flow=Flow(id="inner", nodes=[slow, writer]),
        error_handling=ErrorHandling(timeout_seconds=0.05),
    )
    config = dry_run_config()

    assert node.run({}, config).status == RunnableStatus.FAILURE
    node.dry_run_cleanup(config.dry_run)
    assert Ingest.cleanups == []

    # The timed-out copy is still running in its thread and writes after the holder cleaned up.
    deadline = time.monotonic() + 3
    while not Ingest.cleanups and time.monotonic() < deadline:
        time.sleep(0.02)
    assert Ingest.cleanups == ["writer"]
    assert node._dry_run_flows == []


def test_a_map_keeps_nothing_outside_a_dry_run():
    batch = Map(id="batch", node=Ingest(id="writer"))

    result = batch.run({"input": [{"n": 1}, {"n": 2}]}, RunnableConfig(callbacks=[]))

    assert result.output["output"] == [{"written": True}, {"written": True}]
    assert batch._dry_run_nodes == []


def test_a_map_keeps_no_clone_of_a_node_that_has_nothing_to_clean():
    """A node leaving the base hook alone holds no writes, so retaining one clone per item would
    cost a whole run's worth of nodes for nothing."""
    batch = Map(id="batch", node=Pass(id="passthrough"))
    config = dry_run_config()

    result = batch.run({"input": [{"n": 1}, {"n": 2}, {"n": 3}]}, config)

    assert result.status == RunnableStatus.SUCCESS
    assert batch._dry_run_nodes == []

    # A node that does override it is still kept, so what the clones wrote is still cleaned up.
    writers = Map(id="writers", node=Ingest(id="writer"))
    assert writers.run({"input": [{"n": 1}, {"n": 2}]}, config).status == RunnableStatus.SUCCESS
    assert len(writers._dry_run_nodes) == 2
    writers.dry_run_cleanup(config.dry_run)
    assert len(Ingest.cleanups) == 2 and writers._dry_run_nodes == []
