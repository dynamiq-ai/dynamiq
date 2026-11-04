"""Recover real paused flows after reopening a filesystem checkpoint store."""

from datetime import timedelta
from decimal import Decimal

import pytest

from dynamiq.checkpoints.backends.filesystem import FileSystem
from dynamiq.checkpoints.config import CheckpointBehavior
from dynamiq.checkpoints.types import CheckpointStatus, RunPausedException
from dynamiq.runnables import RunnableStatus
from tests.integration.checkpoints.test_run_pauses import Ledger, at, loan_flow


def recovery_flow(ledger, backend):
    flow = loan_flow(ledger, backend)
    # Successful completion is persisted only when after-node checkpointing is enabled.
    flow.checkpoint.checkpoint_after_node_enabled = True
    return flow


def test_repeated_replace_resume_has_one_current_snapshot_after_restart(tmp_path):
    backend = FileSystem(base_path=str(tmp_path))
    ledger = Ledger()
    flow = recovery_flow(ledger, backend)
    application = {"applicant_id": "app-1", "amount": Decimal("25000.50")}
    flow.run_sync(input_data=application)
    checkpoint_id = flow._checkpoint.id
    resume_at = flow._checkpoint.resume_at
    original_run = flow._checkpoint.run_id

    with at(resume_at - timedelta(hours=1)):
        early = flow.run_sync(input_data=None, resume_from=checkpoint_id)
    assert early.error.type is RunPausedException
    assert len(backend.get_list_by_flow(flow.id)) == 1
    assert backend.get_list_by_run(original_run) == []
    assert backend.load(checkpoint_id).run_id == flow._checkpoint.run_id

    # Rebuild both objects: no in-memory checkpoint or node state can hide a stale file.
    reopened = FileSystem(base_path=str(tmp_path))
    restored_flow = recovery_flow(ledger, reopened)
    with at(resume_at + timedelta(seconds=1)):
        completed = restored_flow.run_sync(input_data=None, resume_from=checkpoint_id)
    assert completed.status == RunnableStatus.SUCCESS
    saved = reopened.load(checkpoint_id)
    assert saved.status == CheckpointStatus.COMPLETED
    assert saved.original_input == application
    assert saved.run_id == restored_flow._checkpoint.run_id
    assert saved.completed_node_ids == restored_flow._checkpoint.completed_node_ids
    assert len(reopened.get_list_by_flow(flow.id)) == 1
    assert ledger.calls == {"credit_score": 1, "disburse": 1}
    assert len(list(tmp_path.rglob("*.json"))) == 1
    assert reopened.delete(checkpoint_id)
    assert FileSystem(base_path=str(tmp_path)).load(checkpoint_id) is None


@pytest.mark.asyncio
async def test_async_pause_resume_and_retention_use_one_persistent_identity(tmp_path):
    backend = FileSystem(base_path=str(tmp_path))
    ledger = Ledger()
    flow = recovery_flow(ledger, backend)
    paused = await flow.run_async(input_data={"applicant_id": "async", "amount": 100})
    assert paused.error.type is RunPausedException
    checkpoint_id, resume_at = flow._checkpoint.id, flow._checkpoint.resume_at
    for hours_early in (2, 1):
        with at(resume_at - timedelta(hours=hours_early)):
            early = await flow.run_async(input_data=None, resume_from=checkpoint_id)
        assert early.error.type is RunPausedException
        saved = await backend.load_async(checkpoint_id)
        assert saved.run_id == flow._checkpoint.run_id
        assert len(await backend.get_list_by_flow_async(flow.id)) == 1

    reopened = FileSystem(base_path=str(tmp_path))
    restored = recovery_flow(ledger, reopened)
    with at(resume_at + timedelta(seconds=1)):
        result = await restored.run_async(input_data=None, resume_from=checkpoint_id)
    assert result.status == RunnableStatus.SUCCESS
    assert (await reopened.load_async(checkpoint_id)).status == CheckpointStatus.COMPLETED
    assert ledger.calls == {"credit_score": 1, "disburse": 1}
    assert await reopened.cleanup_by_flow_async(flow.id, keep_count=0) == 1
    assert await reopened.load_async(checkpoint_id) is None


def test_flow_retention_does_not_delete_a_normalized_alias_waiting_for_input(tmp_path):
    backend = FileSystem(base_path=str(tmp_path))
    first_ledger, second_ledger = Ledger(), Ledger()
    first = loan_flow(first_ledger, backend)
    second = loan_flow(second_ledger, backend)
    first.id, second.id = "customer/a", "customer:a"
    first.run_sync(input_data={"applicant_id": "a", "amount": 100})
    second.run_sync(input_data={"applicant_id": "b", "amount": 200})
    first_id, second_id = first._checkpoint.id, second._checkpoint.id
    first_resume_at = first._checkpoint.resume_at
    assert backend.cleanup_by_flow(second.id, keep_count=0) == 1
    assert backend.load(second_id) is None
    assert backend.load(first_id).original_input["applicant_id"] == "a"

    restored = loan_flow(first_ledger, FileSystem(base_path=str(tmp_path)))
    restored.id = first.id
    with at(first_resume_at + timedelta(seconds=1)):
        completed = restored.run_sync(input_data=None, resume_from=first_id)
    assert completed.status == RunnableStatus.SUCCESS
    assert first_ledger.calls == {"credit_score": 1, "disburse": 1}
    assert second_ledger.calls == {"credit_score": 1}


def test_append_resume_keeps_resumable_history_without_overwriting_ancestors(tmp_path):
    backend = FileSystem(base_path=str(tmp_path))
    ledger = Ledger()
    flow = recovery_flow(ledger, backend)
    flow.checkpoint.behavior = CheckpointBehavior.APPEND
    flow.run_sync(input_data={"applicant_id": "history", "amount": 100})
    paused_id, resume_at = flow._checkpoint.id, flow._checkpoint.resume_at
    paused = backend.load(paused_id)
    with at(resume_at - timedelta(hours=1)):
        flow.run_sync(input_data=None, resume_from=paused_id)
    intermediate_id = flow._checkpoint.id
    assert intermediate_id != paused_id
    assert backend.load(paused_id).model_dump() == paused.model_dump()
    with at(resume_at + timedelta(seconds=1)):
        result = flow.run_sync(input_data=None, resume_from=intermediate_id)
    assert result.status == RunnableStatus.SUCCESS
    chain = FileSystem(base_path=str(tmp_path)).get_chain(flow._checkpoint.id)
    assert chain[0].status == CheckpointStatus.COMPLETED
    identities = [cp.id for cp in chain]
    assert len(identities) == len(set(identities))
    assert identities.index(intermediate_id) < identities.index(paused_id)
    assert chain[-1].original_input == paused.original_input
    assert ledger.calls == {"credit_score": 1, "disburse": 1}
