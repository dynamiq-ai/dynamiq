"""A resumed run keeps the run config the interrupted run was started with.

``Flow`` has always written ``original_config`` into the checkpoint; before this fix nothing
read it back, so a dry run resumed from a mid-flow snapshot without ``mock=`` executed its
remaining mocked nodes for real. The restore mirrors ``original_input``: explicit caller
values win, unset fields fall back to the checkpoint.
"""

from typing import Any, ClassVar, Literal

import pytest
from pydantic import BaseModel

from dynamiq import flows
from dynamiq.checkpoints.backends.in_memory import InMemory
from dynamiq.checkpoints.config import CheckpointBehavior, CheckpointConfig
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.node import Node, NodeDependency
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types.dry_run import DryRunConfig
from dynamiq.types.mocking import MockPolicy, RunMockConfig

EXECUTED: list[str] = []


class StepSchema(BaseModel):
    model_config = {"extra": "allow"}


class StepTool(Node):
    """Records every real execution, so a dry run is provable by absence."""

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "step"
    input_schema: ClassVar[type[StepSchema]] = StepSchema

    def execute(self, input_data: StepSchema, config: RunnableConfig | None = None, **kwargs) -> dict[str, Any]:
        EXECUTED.append(self.id)
        return {"content": f"{self.id} really ran"}


@pytest.fixture(autouse=True)
def clear_executed():
    EXECUTED.clear()
    yield
    EXECUTED.clear()


def build_flow(backend: InMemory) -> flows.Flow:
    first = StepTool(id="first")
    second = StepTool(id="second", depends=[NodeDependency(first)])
    third = StepTool(id="third", depends=[NodeDependency(second)])
    return flows.Flow(
        nodes=[first, second, third],
        checkpoint=CheckpointConfig(
            enabled=True, backend=backend, behavior=CheckpointBehavior.APPEND, max_checkpoints=50
        ),
    )


def mid_flow_snapshot(backend: InMemory, flow: flows.Flow):
    """The APPEND snapshot taken after only the first node completed — a run halted mid-flow."""
    snapshots = backend.get_list_by_flow(flow.id, limit=50)
    partial = [cp for cp in snapshots if cp.completed_node_ids == ["first"]]
    assert partial, f"expected a snapshot with only 'first' complete, got {[c.completed_node_ids for c in snapshots]}"
    return partial[0]


def snapshot_written_during(backend: InMemory, flow: flows.Flow, known_ids: set[str], completed: list[str]):
    """A snapshot appended by the run just made — ids seen beforehand are excluded."""
    fresh = [cp for cp in backend.get_list_by_flow(flow.id, limit=50) if cp.id not in known_ids]
    match = [cp for cp in fresh if cp.completed_node_ids == completed]
    assert match, f"expected a new snapshot completing {completed}, got {[c.completed_node_ids for c in fresh]}"
    return match[0]


class TestResumeRestoresRunConfig:
    def test_a_resumed_dry_run_stays_a_dry_run(self):
        backend = InMemory()
        flow = build_flow(backend)

        flow.run_sync(input_data={}, config=RunnableConfig(mock=RunMockConfig(policy=MockPolicy.ALL)))
        assert EXECUTED == [], "the dry run must not have executed anything"
        checkpoint = mid_flow_snapshot(backend, flow)

        result = flow.run_sync(input_data=None, config=RunnableConfig(), resume_from=checkpoint.id)

        assert result.status == RunnableStatus.SUCCESS
        assert EXECUTED == [], "resuming without mock= must not turn the dry run real"

    def test_an_explicit_resume_config_still_wins(self):
        backend = InMemory()
        flow = build_flow(backend)

        flow.run_sync(input_data={}, config=RunnableConfig(mock=RunMockConfig(policy=MockPolicy.ALL)))
        checkpoint = mid_flow_snapshot(backend, flow)

        result = flow.run_sync(
            input_data=None,
            config=RunnableConfig(mock=RunMockConfig(policy=MockPolicy.NONE)),
            resume_from=checkpoint.id,
        )

        assert result.status == RunnableStatus.SUCCESS
        assert EXECUTED == ["second", "third"], (
            "the caller explicitly asked for a real run: the pending nodes execute; "
            "the already-completed first node is replayed from the checkpoint, not re-run"
        )

    def test_a_plain_run_resumes_with_nothing_restored(self):
        """No mock, no dry_run on the original run: resume behaves exactly as before the fix."""
        backend = InMemory()
        flow = build_flow(backend)

        flow.run_sync(input_data={})
        assert EXECUTED == ["first", "second", "third"]
        checkpoint = mid_flow_snapshot(backend, flow)

        result = flow.run_sync(input_data=None, resume_from=checkpoint.id)

        assert result.status == RunnableStatus.SUCCESS
        assert EXECUTED == ["first", "second", "third", "second", "third"]

    def test_a_snapshot_written_during_a_resume_describes_that_resume(self):
        """Resuming twice: the second resume must inherit the config of the run it resumes.

        A checkpoint appended during a resumed run has to describe that run, not the one that
        first created the checkpoint, or the restore only survives a single hop.
        """
        backend = InMemory()
        flow = build_flow(backend)

        flow.run_sync(input_data={})
        assert EXECUTED == ["first", "second", "third"]
        real_run_snapshot = mid_flow_snapshot(backend, flow)

        known = {cp.id for cp in backend.get_list_by_flow(flow.id, limit=50)}
        EXECUTED.clear()
        flow.run_sync(
            input_data=None,
            config=RunnableConfig(mock=RunMockConfig(policy=MockPolicy.ALL)),
            resume_from=real_run_snapshot.id,
        )
        assert EXECUTED == [], "the resumed run was a dry run"

        dry_run_snapshot = snapshot_written_during(backend, flow, known, ["first", "second"])
        assert dry_run_snapshot.original_config.get("mock"), "the snapshot must record the dry run that wrote it"

        EXECUTED.clear()
        result = flow.run_sync(input_data=None, config=RunnableConfig(), resume_from=dry_run_snapshot.id)

        assert result.status == RunnableStatus.SUCCESS
        assert EXECUTED == [], "resuming a dry run's own snapshot must stay a dry run"

    @pytest.mark.asyncio
    async def test_the_async_path_refreshes_it_too(self):
        """`run_async` has its own checkpoint-init branch and the same gap."""
        backend = InMemory()
        flow = build_flow(backend)

        await flow.run_async(input_data={})
        real_run_snapshot = mid_flow_snapshot(backend, flow)

        known = {cp.id for cp in backend.get_list_by_flow(flow.id, limit=50)}
        EXECUTED.clear()
        await flow.run_async(
            input_data=None,
            config=RunnableConfig(mock=RunMockConfig(policy=MockPolicy.ALL)),
            resume_from=real_run_snapshot.id,
        )
        assert EXECUTED == []

        dry_run_snapshot = snapshot_written_during(backend, flow, known, ["first", "second"])

        EXECUTED.clear()
        await flow.run_async(input_data=None, config=RunnableConfig(), resume_from=dry_run_snapshot.id)

        assert EXECUTED == [], "resuming a dry run's own snapshot must stay a dry run"

    def test_a_real_resume_of_a_dry_runs_snapshot_stays_real(self):
        """Symmetric: a snapshot appended by a MockPolicy.NONE resume must not re-mock the tail."""
        backend = InMemory()
        flow = build_flow(backend)

        flow.run_sync(input_data={}, config=RunnableConfig(mock=RunMockConfig(policy=MockPolicy.ALL)))
        assert EXECUTED == []
        dry_run_snapshot = mid_flow_snapshot(backend, flow)

        known = {cp.id for cp in backend.get_list_by_flow(flow.id, limit=50)}
        flow.run_sync(
            input_data=None,
            config=RunnableConfig(mock=RunMockConfig(policy=MockPolicy.NONE)),
            resume_from=dry_run_snapshot.id,
        )
        assert EXECUTED == ["second", "third"]

        real_run_snapshot = snapshot_written_during(backend, flow, known, ["first", "second"])

        EXECUTED.clear()
        flow.run_sync(input_data=None, config=RunnableConfig(), resume_from=real_run_snapshot.id)

        assert EXECUTED == ["third"], "the tail must not be mocked by the first run's config"

    def test_dry_run_config_is_restored_too(self):
        """The same mechanism covers the RAG-cleanup config, which had the same gap."""
        backend = InMemory()
        flow = build_flow(backend)

        flow.run_sync(
            input_data={},
            config=RunnableConfig(
                mock=RunMockConfig(policy=MockPolicy.ALL),
                dry_run=DryRunConfig(enabled=True, delete_collection=False),
            ),
        )
        checkpoint = mid_flow_snapshot(backend, flow)

        restored = flow._restore_config_from_checkpoint(RunnableConfig(), checkpoint)

        assert restored.mock is not None and restored.mock.policy == MockPolicy.ALL
        assert restored.dry_run is not None
        assert restored.dry_run.enabled is True
        assert restored.dry_run.delete_collection is False
