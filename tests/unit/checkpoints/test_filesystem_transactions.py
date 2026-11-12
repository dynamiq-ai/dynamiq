"""Failure and concurrency boundaries of the persistent checkpoint lifecycle."""

import asyncio
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import timedelta
from threading import Event

import pytest

from dynamiq.checkpoints.backends import filesystem
from dynamiq.checkpoints.backends.filesystem import FileSystem
from dynamiq.checkpoints.checkpoint import FlowCheckpoint
from dynamiq.checkpoints.types import CheckpointStatus


def write_process(root, start, worker):
    backend = FileSystem(base_path=root)
    if not start.wait(20):
        raise RuntimeError("parent did not release the writers")
    for iteration in range(4):
        checkpoint = FlowCheckpoint(
            id="shared",
            flow_id="flow",
            run_id=f"{worker}-{iteration}",
            original_input={"worker": worker, "iteration": iteration},
        )
        backend.save(checkpoint)
        backend.save(checkpoint.model_copy(update={"id": f"unique-{worker}-{iteration}"}))


def interrupted_writer(root, ready, release):
    def wait_before_replace(source, destination):
        ready.set()
        if not release.wait(30):
            raise RuntimeError("writer was not terminated")

    filesystem.os.replace = wait_before_replace
    backend = FileSystem(base_path=root)
    checkpoint = backend.load("stable")
    checkpoint.run_id = "interrupted-resume"
    backend.save(checkpoint)


@pytest.fixture
def backend(tmp_path):
    return FileSystem(base_path=str(tmp_path))


@pytest.mark.parametrize("failure", ["serialization", "flush", "replace"])
def test_failed_upsert_preserves_last_committed_snapshot(backend, tmp_path, monkeypatch, failure):
    original = FlowCheckpoint(id="stable", flow_id="flow", run_id="old", original_input="committed")
    backend.save(original)
    updated = original.model_copy(update={"run_id": "resumed", "original_input": "uncommitted"})

    def fail(*args, **kwargs):
        raise OSError("injected write failure")

    with monkeypatch.context() as patch:
        if failure == "serialization":
            patch.setattr(filesystem.json, "dump", fail)
        elif failure == "flush":
            patch.setattr(filesystem.os, "fsync", fail)
        else:
            patch.setattr(filesystem.os, "replace", fail)
        with pytest.raises(OSError, match="injected write failure"):
            backend.save(updated)

    assert backend.load(original.id).model_dump() == original.model_dump()
    assert len(backend.get_list_by_flow("flow")) == 1
    assert not list(tmp_path.rglob("*.tmp"))
    backend.update(updated)
    assert backend.load(original.id).original_input == "uncommitted"


def test_cancelled_write_removes_temporary_file_and_releases_lock(backend, tmp_path, monkeypatch):
    checkpoint = FlowCheckpoint(flow_id="flow", run_id="run")

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt()

    with monkeypatch.context() as patch:
        patch.setattr(filesystem.json, "dump", interrupt)
        with pytest.raises(KeyboardInterrupt):
            backend.save(checkpoint)
    assert backend.load(checkpoint.id) is None
    assert not list(tmp_path.rglob("*.tmp"))
    backend.save(checkpoint)
    assert backend.load(checkpoint.id).id == checkpoint.id


def test_separate_instances_serialize_upserts_before_discovering_existing_record(tmp_path, monkeypatch):
    first = FileSystem(base_path=str(tmp_path))
    second = FileSystem(base_path=str(tmp_path))
    old = FlowCheckpoint(id="same", flow_id="flow", run_id="old")
    new = old.model_copy(update={"run_id": "new", "original_input": "last writer"})
    replacing = Event()
    release = Event()
    second_started = Event()
    real_replace = filesystem.os.replace

    def pause_first(source, destination):
        if not replacing.is_set():
            replacing.set()
            assert release.wait(10)
        return real_replace(source, destination)

    def save_second():
        second_started.set()
        return second.save(new)

    monkeypatch.setattr(filesystem.os, "replace", pause_first)
    with ThreadPoolExecutor(max_workers=2) as workers:
        pending_first = workers.submit(first.save, old)
        try:
            assert replacing.wait(10)
            pending_second = workers.submit(save_second)
            assert second_started.wait(10)
            with pytest.raises(TimeoutError):
                pending_second.result(timeout=0.1)
        finally:
            release.set()
        assert pending_first.result(timeout=10) == old.id
        assert pending_second.result(timeout=10) == old.id
    assert first.load(old.id).original_input == "last writer"
    assert len(first.get_list_by_flow("flow")) == 1
    assert len(list(tmp_path.rglob("*.json"))) == 1


@pytest.mark.parametrize("operation", ["load", "delete", "list", "cleanup"])
def test_readers_and_removers_wait_for_in_flight_publication(backend, tmp_path, monkeypatch, operation):
    checkpoint = FlowCheckpoint(id="same", flow_id="flow", run_id="run")
    observer = FileSystem(base_path=str(tmp_path))
    replacing, release, observing = Event(), Event(), Event()
    real_replace = filesystem.os.replace

    def pause_replace(source, destination):
        replacing.set()
        assert release.wait(10)
        return real_replace(source, destination)

    def observe():
        observing.set()
        if operation == "list":
            return observer.get_list_by_flow("flow")
        if operation == "cleanup":
            return observer.cleanup_by_flow("flow", keep_count=0)
        return getattr(observer, operation)(checkpoint.id)

    monkeypatch.setattr(filesystem.os, "replace", pause_replace)
    with ThreadPoolExecutor(max_workers=2) as workers:
        writer = workers.submit(backend.save, checkpoint)
        try:
            assert replacing.wait(10)
            reader = workers.submit(observe)
            assert observing.wait(10)
            with pytest.raises(TimeoutError):
                reader.result(timeout=0.1)
        finally:
            release.set()
        writer.result(timeout=10)
        result = reader.result(timeout=10)
    if operation == "load":
        assert result.id == checkpoint.id
    elif operation == "list":
        assert [item.id for item in result] == [checkpoint.id]
    else:
        assert result == 1
        assert backend.load(checkpoint.id) is None


def test_process_writers_do_not_leave_resume_copies(tmp_path):
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    workers = [context.Process(target=write_process, args=(str(tmp_path), start, worker)) for worker in range(3)]
    try:
        for worker in workers:
            worker.start()
        start.set()
        for worker in workers:
            worker.join(timeout=30)
            assert worker.exitcode == 0
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            if worker.pid is not None:
                worker.join(timeout=10)
    backend = FileSystem(base_path=str(tmp_path))
    checkpoints = backend.get_list_by_flow("flow")
    assert len(checkpoints) == 13
    assert len({checkpoint.id for checkpoint in checkpoints}) == 13
    shared = backend.load("shared")
    assert shared.original_input["iteration"] == 3
    assert len(list(tmp_path.rglob("*.json"))) == 13


def test_terminated_writer_releases_lock_without_publishing_partial_state(backend, tmp_path):
    checkpoint = FlowCheckpoint(id="stable", flow_id="flow", run_id="committed")
    backend.save(checkpoint)
    context = multiprocessing.get_context("spawn")
    ready, release = context.Event(), context.Event()
    worker = context.Process(target=interrupted_writer, args=(str(tmp_path), ready, release))
    try:
        worker.start()
        assert ready.wait(20)
        worker.terminate()
        worker.join(timeout=10)
        assert not worker.is_alive()
    finally:
        if worker.is_alive():
            worker.kill()
        worker.join(timeout=10)
    # OS ownership ends with the process. Abandoned temporary files are never
    # treated as snapshots, and a fresh instance can still replace the record.
    reopened = FileSystem(base_path=str(tmp_path))
    assert reopened.load(checkpoint.id).run_id == "committed"
    reopened.update(checkpoint.model_copy(update={"run_id": "recovered"}))
    assert reopened.load(checkpoint.id).run_id == "recovered"
    assert len(reopened.get_list_by_flow("flow")) == 1


@pytest.mark.asyncio
async def test_async_resume_upserts_share_sync_storage_contract(backend):
    original = FlowCheckpoint(id="shared", flow_id="flow", run_id="first")
    await backend.save_async(original)
    updates = [original.model_copy(update={"run_id": f"resume-{i}", "original_input": i}) for i in range(8)]
    await asyncio.gather(*(backend.save_async(checkpoint) for checkpoint in updates))
    records = await backend.get_list_by_flow_async("flow")
    assert len(records) == 1
    loaded = await backend.load_async(original.id)
    assert loaded.run_id == f"resume-{loaded.original_input}"
    assert await backend.get_list_by_run_async("first") == []
    assert await backend.delete_async(original.id)
    assert await backend.load_async(original.id) is None


def test_retention_keeps_waiting_checkpoint_until_continued(backend):
    waiting = FlowCheckpoint(flow_id="flow", run_id="waiting", status=CheckpointStatus.PENDING_INPUT)
    active = FlowCheckpoint(flow_id="flow", run_id="active", created_at=waiting.created_at + timedelta(seconds=1))
    backend.save(waiting)
    backend.save(active)
    assert backend.cleanup_by_flow("flow", keep_count=1) == 0
    continuation = FlowCheckpoint(
        flow_id="flow",
        run_id="continued",
        parent_checkpoint_id=waiting.id,
        created_at=active.created_at + timedelta(seconds=1),
    )
    backend.save(continuation)
    assert backend.cleanup_by_flow("flow", keep_count=1) == 2
    assert [checkpoint.id for checkpoint in backend.get_list_by_flow("flow")] == [continuation.id]


def test_retention_expires_waiting_snapshot_and_honors_fetch_bound(backend):
    old = FlowCheckpoint(flow_id="flow", run_id="old", status=CheckpointStatus.PAUSED)
    old.created_at -= timedelta(hours=2)
    backend.save(old)
    assert backend.cleanup_by_flow("flow", max_ttl_minutes=30) == 1
    for index in range(4):
        backend.save(FlowCheckpoint(flow_id="flow", run_id=str(index)))
    backend.max_cleanup_results = 2
    assert backend.cleanup_by_flow("flow", keep_count=0) == 2
    assert len(backend.get_list_by_flow("flow")) == 2
