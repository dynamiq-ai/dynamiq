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
