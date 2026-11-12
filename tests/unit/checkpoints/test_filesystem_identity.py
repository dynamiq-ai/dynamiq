"""Filesystem checkpoints must preserve logical identity across their whole lifecycle."""

import json
import re
from datetime import timedelta
from decimal import Decimal
from io import BytesIO
from pathlib import Path

import pytest

from dynamiq.checkpoints.backends.filesystem import FileSystem
from dynamiq.checkpoints.checkpoint import FlowCheckpoint
from dynamiq.checkpoints.types import CheckpointStatus
from dynamiq.utils import encode_reversible


@pytest.fixture
def backend(tmp_path):
    return FileSystem(base_path=str(tmp_path))


def legacy_file(root: Path, checkpoint: FlowCheckpoint) -> Path:
    """Write the pre-fix layout without calling the implementation's path helpers."""

    def sanitize(value):
        return re.sub(r"[^a-zA-Z0-9_\-]", "_", value)

    stamp = checkpoint.created_at.strftime("%Y-%m-%dT%H-%M-%S-%f")[:-4]
    folder = root / sanitize(checkpoint.flow_id) / f"{stamp}__{sanitize(checkpoint.wf_run_id or checkpoint.run_id)}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stamp}__{sanitize(checkpoint.id)}.json"
    path.write_text(json.dumps(checkpoint.to_dict(), default=encode_reversible), encoding="utf-8")
    return path


@pytest.mark.parametrize("other", ["customer:a", "customer_a"])
def test_flow_listing_never_includes_a_normalized_alias(backend, other):
    first = FlowCheckpoint(flow_id="customer/a", run_id="run")
    second = FlowCheckpoint(flow_id=other, run_id="run")
    backend.save(first)
    backend.save(second)
    assert [cp.id for cp in backend.get_list_by_flow("customer/a")] == [first.id]
    assert [cp.id for cp in backend.get_list_by_flow(other)] == [second.id]


@pytest.mark.parametrize("method", ["load", "delete"])
def test_missing_checkpoint_alias_does_not_resolve_another_record(backend, method):
    checkpoint = FlowCheckpoint(id="pause/a", flow_id="flow", run_id="run")
    backend.save(checkpoint)
    assert getattr(backend, method)("pause:a") in (None, False)
    assert backend.load("pause/a").id == "pause/a"


def test_colliding_checkpoint_ids_survive_same_timestamp_and_run(backend):
    first = FlowCheckpoint(id="pause/a", flow_id="flow", run_id="run", original_input="first")
    second = first.model_copy(update={"id": "pause:a", "original_input": "second"})
    backend.save(first)
    backend.save(second)
    assert backend.load(first.id).original_input == "first"
    assert backend.load(second.id).original_input == "second"
    assert {cp.id for cp in backend.get_list_by_flow("flow")} == {first.id, second.id}


@pytest.mark.parametrize("method", ["save", "update"])
def test_upsert_keeps_one_record_when_run_and_timestamp_change(backend, method):
    checkpoint = FlowCheckpoint(id="checkpoint", flow_id="flow", run_id="first")
    backend.save(checkpoint)
    resumed = checkpoint.model_copy(
        update={
            "run_id": "resumed",
            "wf_run_id": "workflow-resumed",
            "created_at": checkpoint.created_at + timedelta(seconds=1),
            "status": CheckpointStatus.COMPLETED,
            "original_input": "new input",
        }
    )
    getattr(backend, method)(resumed)
    loaded = backend.load(checkpoint.id)
    assert loaded.status == CheckpointStatus.COMPLETED
    assert loaded.original_input == "new input"
    assert len(backend.get_list_by_flow("flow")) == 1
    assert backend.get_list_by_run("first") == []
    assert [cp.id for cp in backend.get_list_by_run("workflow-resumed")] == [checkpoint.id]
    assert backend.delete(checkpoint.id)
    assert backend.load(checkpoint.id) is None


def test_cleanup_is_scoped_to_exact_flow_identity(backend):
    keep = FlowCheckpoint(flow_id="customer/a", run_id="run")
    remove = FlowCheckpoint(flow_id="customer:a", run_id="run")
    backend.save(keep)
    backend.save(remove)
    assert backend.cleanup_by_flow(remove.flow_id, keep_count=0) == 1
    assert backend.load(keep.id) is not None
    assert backend.load(remove.id) is None


def test_legacy_reads_verify_the_payload_identity(backend, tmp_path):
    checkpoint = FlowCheckpoint(id="pause/a", flow_id="customer/a", run_id="run")
    legacy_file(tmp_path, checkpoint)
    assert backend.load(checkpoint.id).flow_id == checkpoint.flow_id
    assert backend.load("pause:a") is None
    assert backend.get_list_by_flow("customer:a") == []
    assert not backend.delete("pause:a")
    assert backend.load(checkpoint.id) is not None


def test_legacy_record_is_updated_in_place(backend, tmp_path):
    checkpoint = FlowCheckpoint(id="pause/a", flow_id="customer/a", run_id="old")
    original_path = legacy_file(tmp_path, checkpoint)
    resumed = checkpoint.model_copy(update={"run_id": "new", "original_input": "resumed"})
    backend.save(resumed)
    assert backend.load(checkpoint.id).original_input == "resumed"
    assert list(tmp_path.rglob("*.json")) == [original_path]


def test_ambiguous_legacy_snapshots_do_not_resume_arbitrary_state(backend, tmp_path):
    checkpoint = FlowCheckpoint(id="same", flow_id="flow", run_id="old")
    legacy_file(tmp_path, checkpoint)
    newer = checkpoint.model_copy(update={"run_id": "new", "original_input": "different"})
    legacy_file(tmp_path, newer)
    with pytest.raises(ValueError, match="[Aa]mbiguous|[Dd]uplicate"):
        backend.load(checkpoint.id)


def test_authoritative_save_consolidates_legacy_duplicates(backend, tmp_path):
    checkpoint = FlowCheckpoint(id="same", flow_id="flow", run_id="old")
    legacy_file(tmp_path, checkpoint)
    legacy_file(tmp_path, checkpoint.model_copy(update={"run_id": "new"}))
    authoritative = checkpoint.model_copy(update={"original_input": "repaired"})
    backend.save(authoritative)
    assert backend.load(checkpoint.id).original_input == "repaired"
    assert len(backend.get_list_by_flow("flow")) == 1
    assert len(list(tmp_path.rglob("*.json"))) == 1


def test_corrupt_matching_snapshot_is_not_reported_missing(backend, tmp_path):
    checkpoint = FlowCheckpoint(id="broken", flow_id="flow", run_id="run")
    path = legacy_file(tmp_path, checkpoint)
    path.write_text('{"id":', encoding="utf-8")
    with pytest.raises(ValueError):
        backend.load(checkpoint.id)


def test_long_unicode_identifiers_round_trip_without_filename_overflow(backend):
    checkpoint = FlowCheckpoint(id="暂停" * 180, flow_id="工作流" * 180, run_id="运行" * 180)
    backend.save(checkpoint)
    assert backend.load(checkpoint.id).model_dump() == checkpoint.model_dump()
    assert [cp.id for cp in backend.get_list_by_flow(checkpoint.flow_id)] == [checkpoint.id]


@pytest.mark.parametrize("query", ["flow", "status", "before", "run"])
def test_filtered_listing_rejects_conflicting_legacy_versions(backend, tmp_path, query):
    old = FlowCheckpoint(id="same", flow_id="old-flow", run_id="old-run", status=CheckpointStatus.PAUSED)
    new = old.model_copy(
        update={
            "flow_id": "new-flow",
            "run_id": "new-run",
            "status": CheckpointStatus.COMPLETED,
            "created_at": old.created_at + timedelta(seconds=1),
        }
    )
    legacy_file(tmp_path, old)
    legacy_file(tmp_path, new)
    with pytest.raises(ValueError, match="[Aa]mbiguous|[Dd]uplicate"):
        if query == "run":
            backend.get_list_by_run(old.run_id)
        elif query == "before":
            backend.get_list_by_flow(old.flow_id, before=new.created_at)
        elif query == "status":
            backend.get_latest_by_flow(old.flow_id, status=CheckpointStatus.PAUSED)
        else:
            backend.get_list_by_flow(old.flow_id)


def test_new_and_legacy_records_can_coexist_without_aliasing(backend, tmp_path):
    old = FlowCheckpoint(id="pause/a", flow_id="flow/a", run_id="run/a")
    legacy_path = legacy_file(tmp_path, old)
    new = old.model_copy(update={"id": "pause:a", "flow_id": "flow:a", "run_id": "run:a"})
    backend.save(new)
    backend.update(old.model_copy(update={"original_input": "updated legacy"}))
    assert legacy_path.exists()
    assert backend.load(old.id).original_input == "updated legacy"
    assert backend.load(new.id).flow_id == new.flow_id
    assert [cp.id for cp in backend.get_list_by_flow(new.flow_id)] == [new.id]
    assert backend.delete(new.id)
    assert backend.load(old.id).original_input == "updated legacy"


def test_delete_consolidated_identity_removes_every_legacy_copy_only(backend, tmp_path):
    checkpoint = FlowCheckpoint(id="pause/a", flow_id="flow", run_id="first")
    legacy_file(tmp_path, checkpoint)
    legacy_file(tmp_path, checkpoint.model_copy(update={"run_id": "second"}))
    other = checkpoint.model_copy(update={"id": "pause:a", "run_id": "other"})
    legacy_file(tmp_path, other)
    assert backend.delete(checkpoint.id)
    assert backend.load(checkpoint.id) is None
    assert backend.load(other.id).run_id == "other"
    assert not backend.delete(checkpoint.id)


def test_legacy_alias_update_cannot_create_or_modify_a_different_checkpoint(backend, tmp_path):
    checkpoint = FlowCheckpoint(id="pause/a", flow_id="flow", run_id="run")
    legacy_file(tmp_path, checkpoint)
    alias = checkpoint.model_copy(update={"id": "pause:a", "original_input": "overwrite attempt"})
    with pytest.raises(FileNotFoundError):
        backend.update(alias)
    assert backend.load(checkpoint.id).original_input is None
    assert len(list(tmp_path.rglob("*.json"))) == 1


def test_payload_scope_changes_are_visible_without_moving_the_file(backend, tmp_path):
    checkpoint = FlowCheckpoint(id="same", flow_id="old-flow", run_id="old-run")
    backend.save(checkpoint)
    path = next(tmp_path.rglob("*.json"))
    changed = checkpoint.model_copy(update={"flow_id": "new-flow", "run_id": "new-run", "wf_run_id": "wf"})
    backend.update(changed)
    assert list(tmp_path.rglob("*.json")) == [path]
    assert backend.get_list_by_flow("old-flow") == []
    assert backend.get_list_by_run("old-run") == []
    assert backend.get_list_by_flow_and_run("new-flow", "new-run")[0].id == checkpoint.id
    assert backend.get_latest_by_flow_and_run("new-flow", "absent", wf_run_id="wf").id == checkpoint.id
    assert backend.cleanup_by_flow("old-flow", keep_count=0) == 0
    assert backend.cleanup_by_flow("new-flow", keep_count=0) == 1


@pytest.mark.parametrize("identifiers", [("Case", "case"), ("é", "e\u0301"), (".", ".."), ("", "_")])
def test_distinct_identifiers_have_distinct_physical_records(backend, identifiers):
    first, second = identifiers
    original = FlowCheckpoint(id=first, flow_id=first, run_id=first)
    other = original.model_copy(update={"id": second, "flow_id": second, "run_id": second})
    backend.save(original)
    backend.save(other)
    assert backend.load(first).flow_id == first
    assert backend.load(second).flow_id == second
    assert backend.get_list_by_flow(first)[0].id == first
    assert backend.get_list_by_flow(second)[0].id == second
    assert backend.delete(first)
    assert backend.load(second).id == second


def test_reversible_state_survives_legacy_migration_and_upsert(backend, tmp_path):
    checkpoint = FlowCheckpoint(
        id="binary",
        flow_id="flow",
        run_id="old",
        original_input={"amount": Decimal("19.35"), "file": BytesIO(b"\0\xffpayload")},
    )
    legacy_file(tmp_path, checkpoint)
    resumed = backend.load(checkpoint.id)
    assert resumed.original_input["amount"] == Decimal("19.35")
    assert resumed.original_input["file"].getvalue() == b"\0\xffpayload"
    resumed.run_id = "new"
    backend.update(resumed)
    restored = backend.load(checkpoint.id)
    assert restored.original_input["file"].getvalue() == b"\0\xffpayload"
    assert len(backend.get_list_by_flow("flow")) == 1


def test_chain_uses_exact_checkpoint_ids_in_mixed_layouts(backend, tmp_path):
    parent = FlowCheckpoint(id="parent/a", flow_id="flow", run_id="old")
    legacy_file(tmp_path, parent)
    alias = parent.model_copy(update={"id": "parent:a", "run_id": "unrelated"})
    backend.save(alias)
    child = FlowCheckpoint(id="child", flow_id="flow", run_id="new", parent_checkpoint_id=parent.id)
    backend.save(child)
    assert [cp.id for cp in backend.get_chain(child.id)] == [child.id, parent.id]
    parent.parent_checkpoint_id = child.id
    backend.update(parent)
    assert [cp.id for cp in backend.get_chain(child.id)] == [child.id, parent.id]


@pytest.mark.parametrize("payload", ["null", "[]", '{"id":"bad"}', '{"id":'])
def test_invalid_matching_record_is_an_error_for_all_identity_mutations(backend, tmp_path, payload):
    checkpoint = FlowCheckpoint(id="bad", flow_id="flow", run_id="run")
    path = legacy_file(tmp_path, checkpoint)
    path.write_text(payload, encoding="utf-8")
    for operation, argument in ((backend.load, "bad"), (backend.delete, "bad"), (backend.update, checkpoint)):
        with pytest.raises(ValueError, match="Invalid checkpoint"):
            operation(argument)
    assert path.read_text(encoding="utf-8") == payload


def test_failed_legacy_consolidation_never_deletes_the_previous_snapshots(backend, tmp_path, monkeypatch):
    checkpoint = FlowCheckpoint(id="same", flow_id="flow", run_id="first")
    first = legacy_file(tmp_path, checkpoint)
    second = legacy_file(tmp_path, checkpoint.model_copy(update={"run_id": "second"}))
    contents = {path: path.read_bytes() for path in (first, second)}

    def fail_replace(*args):
        raise OSError("publication failed")

    monkeypatch.setattr("dynamiq.checkpoints.backends.filesystem.os.replace", fail_replace)
    with pytest.raises(OSError, match="publication failed"):
        backend.save(checkpoint.model_copy(update={"original_input": "new"}))
    assert {path: path.read_bytes() for path in (first, second)} == contents
    with pytest.raises(ValueError, match="Ambiguous"):
        backend.load(checkpoint.id)


def test_compound_query_ignores_ambiguity_in_an_unrelated_run(backend, tmp_path):
    healthy = FlowCheckpoint(id="healthy", flow_id="flow", run_id="requested")
    backend.save(healthy)
    duplicate = FlowCheckpoint(id="duplicate", flow_id="flow", run_id="unrelated")
    legacy_file(tmp_path, duplicate)
    legacy_file(tmp_path, duplicate.model_copy(update={"run_id": "unrelated-resume"}))
    assert [cp.id for cp in backend.get_list_by_flow_and_run("flow", "requested")] == [healthy.id]
    assert backend.get_latest_by_flow_and_run("flow", "requested").id == healthy.id
    with pytest.raises(ValueError, match="Ambiguous"):
        backend.get_list_by_flow_and_run("flow", "unrelated")


def test_latest_never_falls_back_to_stale_state_when_newer_snapshot_is_corrupt(backend, tmp_path):
    old = FlowCheckpoint(id="old", flow_id="flow", run_id="run")
    backend.save(old)
    new = old.model_copy(update={"id": "new", "created_at": old.created_at + timedelta(seconds=1)})
    path = legacy_file(tmp_path, new)
    path.write_text('{"id":', encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid checkpoint"):
        backend.get_latest_by_flow("flow")
    with pytest.raises(ValueError, match="Invalid checkpoint"):
        backend.cleanup_by_flow("flow", keep_count=0)
    assert backend.load(old.id) is not None
    assert path.exists()


@pytest.mark.parametrize("missing", ["id", "flow_id", "run_id", "created_at", "updated_at", "status"])
def test_persisted_core_fields_cannot_be_recreated_from_model_defaults(backend, tmp_path, missing):
    checkpoint = FlowCheckpoint(id="stable", flow_id="flow", run_id="run")
    path = legacy_file(tmp_path, checkpoint)
    data = json.loads(path.read_text(encoding="utf-8"))
    del data[missing]
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid checkpoint"):
        backend.load(checkpoint.id)
    with pytest.raises(ValueError, match="Invalid checkpoint"):
        backend.get_list_by_flow("flow")
    with pytest.raises(ValueError, match="Invalid checkpoint"):
        backend.delete(checkpoint.id)
    assert path.exists()
