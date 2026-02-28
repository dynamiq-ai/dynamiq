"""Unit tests for checkpoint storage backends (InMemory, FileSystem)."""

import time
from pathlib import Path

import pytest

from dynamiq.checkpoints.backends.base import CheckpointBackend
from dynamiq.checkpoints.backends.filesystem import FileSystem
from dynamiq.checkpoints.backends.in_memory import InMemory
from dynamiq.checkpoints.checkpoint import CheckpointStatus, FlowCheckpoint, NodeCheckpointState


class BackendTestMixin:
    """Shared tests for all checkpoint backends."""

    @pytest.fixture
    def backend(self) -> CheckpointBackend:
        """Override in subclasses to provide backend instance."""
        raise NotImplementedError

    @pytest.fixture
    def sample_checkpoint(self) -> FlowCheckpoint:
        """Create a sample checkpoint for testing."""
        return FlowCheckpoint(
            flow_id="test-flow",
            run_id="run-123",
            original_input={"query": "test"},
            metadata={"test_key": "test_value"},
        )

    def test_save_and_load(self, backend: CheckpointBackend, sample_checkpoint: FlowCheckpoint):
        """Test saving and loading a checkpoint."""
        checkpoint_id = backend.save(sample_checkpoint)

        loaded = backend.load(checkpoint_id)

        assert loaded is not None
        assert loaded.id == sample_checkpoint.id
        assert loaded.flow_id == sample_checkpoint.flow_id
        assert loaded.run_id == sample_checkpoint.run_id
        assert loaded.original_input == sample_checkpoint.original_input

    def test_load_nonexistent(self, backend: CheckpointBackend):
        """Test loading a non-existent checkpoint returns None."""
        result = backend.load("nonexistent-id")
        assert result is None

    def test_update(self, backend: CheckpointBackend, sample_checkpoint: FlowCheckpoint):
        """Test updating an existing checkpoint."""
        backend.save(sample_checkpoint)

        node_state = NodeCheckpointState(
            node_id="node-1",
            node_type="TestNode",
            status="success",
            output_data={"result": "done"},
        )
        sample_checkpoint.mark_node_complete("node-1", node_state)
        sample_checkpoint.status = CheckpointStatus.COMPLETED

        backend.update(sample_checkpoint)

        loaded = backend.load(sample_checkpoint.id)
        assert loaded is not None
        assert loaded.status == CheckpointStatus.COMPLETED
        assert "node-1" in loaded.node_states
        assert "node-1" in loaded.completed_node_ids

    def test_delete(self, backend: CheckpointBackend, sample_checkpoint: FlowCheckpoint):
        """Test deleting a checkpoint."""
        backend.save(sample_checkpoint)

        deleted = backend.delete(sample_checkpoint.id)
        assert deleted is True

        loaded = backend.load(sample_checkpoint.id)
        assert loaded is None

    def test_delete_nonexistent(self, backend: CheckpointBackend):
        """Test deleting a non-existent checkpoint returns False."""
        result = backend.delete("nonexistent-id")
        assert result is False

    def test_list_by_flow(self, backend: CheckpointBackend):
        """Test listing checkpoints for a flow."""
        flow_id = "list-test-flow"

        for i in range(5):
            checkpoint = FlowCheckpoint(flow_id=flow_id, run_id=f"run-{i}")
            backend.save(checkpoint)
            time.sleep(0.01)

        checkpoints = backend.get_list_by_flow(flow_id, limit=10)

        assert len(checkpoints) == 5
        for i in range(len(checkpoints) - 1):
            assert checkpoints[i].created_at >= checkpoints[i + 1].created_at

    def test_list_by_flow_with_limit(self, backend: CheckpointBackend):
        """Test listing checkpoints with limit."""
        flow_id = "limit-test-flow"

        for i in range(5):
            checkpoint = FlowCheckpoint(flow_id=flow_id, run_id=f"run-{i}")
            backend.save(checkpoint)

        checkpoints = backend.get_list_by_flow(flow_id, limit=3)
        assert len(checkpoints) == 3

    def test_list_by_flow_with_status_filter(self, backend: CheckpointBackend):
        """Test listing checkpoints with status filter."""
        flow_id = "status-test-flow"

        cp1 = FlowCheckpoint(flow_id=flow_id, run_id="run-1", status=CheckpointStatus.ACTIVE)
        cp2 = FlowCheckpoint(flow_id=flow_id, run_id="run-2", status=CheckpointStatus.COMPLETED)
        cp3 = FlowCheckpoint(flow_id=flow_id, run_id="run-3", status=CheckpointStatus.FAILED)

        backend.save(cp1)
        backend.save(cp2)
        backend.save(cp3)

        active_checkpoints = backend.get_list_by_flow(flow_id, status=CheckpointStatus.ACTIVE)
        assert len(active_checkpoints) == 1
        assert active_checkpoints[0].status == CheckpointStatus.ACTIVE

        completed_checkpoints = backend.get_list_by_flow(flow_id, status=CheckpointStatus.COMPLETED)
        assert len(completed_checkpoints) == 1

    def test_list_by_flow_empty(self, backend: CheckpointBackend):
        """Test listing checkpoints for a flow with no checkpoints."""
        checkpoints = backend.get_list_by_flow("nonexistent-flow")
        assert checkpoints == []

    def test_get_latest(self, backend: CheckpointBackend):
        """Test getting the most recent checkpoint."""
        flow_id = "latest-test-flow"

        cp1 = FlowCheckpoint(flow_id=flow_id, run_id="run-1")
        backend.save(cp1)
        time.sleep(0.01)

        cp2 = FlowCheckpoint(flow_id=flow_id, run_id="run-2")
        backend.save(cp2)
        time.sleep(0.01)

        cp3 = FlowCheckpoint(flow_id=flow_id, run_id="run-3")
        backend.save(cp3)

        latest = backend.get_latest_by_flow(flow_id)
        assert latest is not None
        assert latest.run_id == "run-3"

    def test_get_latest_with_status_filter(self, backend: CheckpointBackend):
        """Test getting the most recent checkpoint with status filter."""
        flow_id = "latest-status-test-flow"

        cp1 = FlowCheckpoint(flow_id=flow_id, run_id="run-1", status=CheckpointStatus.COMPLETED)
        backend.save(cp1)
        time.sleep(0.01)

        cp2 = FlowCheckpoint(flow_id=flow_id, run_id="run-2", status=CheckpointStatus.ACTIVE)
        backend.save(cp2)

        latest_completed = backend.get_latest_by_flow(flow_id, status=CheckpointStatus.COMPLETED)
        assert latest_completed is not None
        assert latest_completed.run_id == "run-1"
        assert latest_completed.status == CheckpointStatus.COMPLETED

    def test_get_latest_nonexistent(self, backend: CheckpointBackend):
        """Test getting latest for a flow with no checkpoints."""
        result = backend.get_latest_by_flow("nonexistent-flow")
        assert result is None

    def test_cleanup_by_count(self, backend: CheckpointBackend):
        """Test cleanup keeps specified number of recent checkpoints."""
        flow_id = "cleanup-test-flow"

        for i in range(10):
            checkpoint = FlowCheckpoint(flow_id=flow_id, run_id=f"run-{i}")
            backend.save(checkpoint)
            time.sleep(0.01)

        deleted = backend.cleanup_by_flow(flow_id, keep_count=3)

        assert deleted == 7
        remaining = backend.get_list_by_flow(flow_id, limit=100)
        assert len(remaining) == 3

    def test_checkpoint_with_complex_data(self, backend: CheckpointBackend):
        """Test saving and loading checkpoint with complex nested data."""
        checkpoint = FlowCheckpoint(
            flow_id="complex-flow",
            run_id="run-complex",
            original_input={
                "query": "test",
                "nested": {"key": "value", "list": [1, 2, 3]},
                "numbers": [1.5, 2.5],
            },
            metadata={
                "tags": ["tag1", "tag2"],
                "config": {"option1": True, "option2": 42},
            },
        )

        node_state = NodeCheckpointState(
            node_id="agent-1",
            node_type="Agent",
            status="success",
            input_data={"messages": [{"role": "user", "content": "Hello"}]},
            output_data={"response": "Hi there!"},
            internal_state={
                "conversation_history": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ],
                "tool_calls": 3,
            },
        )
        checkpoint.node_states["agent-1"] = node_state
        checkpoint.completed_node_ids.append("agent-1")

        backend.save(checkpoint)

        loaded = backend.load(checkpoint.id)

        assert loaded is not None
        assert loaded.original_input["nested"]["list"] == [1, 2, 3]
        assert loaded.metadata["tags"] == ["tag1", "tag2"]
        assert loaded.node_states["agent-1"].internal_state["tool_calls"] == 3

    def test_get_list_by_run_matches_wf_run_id(self, backend: CheckpointBackend):
        """get_list_by_run matches checkpoints by wf_run_id too."""
        wf_run_id = "shared-workflow-run"
        cp1 = FlowCheckpoint(flow_id="flow-1", run_id="run-a", wf_run_id=wf_run_id)
        cp2 = FlowCheckpoint(flow_id="flow-1", run_id="run-b", wf_run_id=wf_run_id)
        cp3 = FlowCheckpoint(flow_id="flow-1", run_id="run-c", wf_run_id="other-wf-run")

        backend.save(cp1)
        backend.save(cp2)
        backend.save(cp3)

        results = backend.get_list_by_run(wf_run_id)
        assert len(results) == 2
        assert {r.run_id for r in results} == {"run-a", "run-b"}

    def test_get_list_by_flow_and_run(self, backend: CheckpointBackend):
        """get_list_by_flow_and_run returns checkpoints matching both flow_id and run_id."""
        flow_id = "target-flow"
        wf_run_id = "target-wf-run"
        cp1 = FlowCheckpoint(flow_id=flow_id, run_id="run-1", wf_run_id=wf_run_id)
        cp2 = FlowCheckpoint(flow_id=flow_id, run_id="run-2", wf_run_id=wf_run_id)
        cp3 = FlowCheckpoint(flow_id="other-flow", run_id="run-3", wf_run_id=wf_run_id)
        cp4 = FlowCheckpoint(flow_id=flow_id, run_id="run-4", wf_run_id="other-wf-run")

        for cp in [cp1, cp2, cp3, cp4]:
            backend.save(cp)
            time.sleep(0.01)

        results = backend.get_list_by_flow_and_run(flow_id, wf_run_id)
        assert len(results) == 2
        assert {r.run_id for r in results} == {"run-1", "run-2"}

    def test_get_list_by_flow_and_run_with_status(self, backend: CheckpointBackend):
        """get_list_by_flow_and_run filters by status."""
        flow_id = "status-flow"
        wf_run_id = "status-wf-run"
        cp1 = FlowCheckpoint(flow_id=flow_id, run_id="run-1", wf_run_id=wf_run_id, status=CheckpointStatus.COMPLETED)
        cp2 = FlowCheckpoint(flow_id=flow_id, run_id="run-2", wf_run_id=wf_run_id, status=CheckpointStatus.ACTIVE)

        backend.save(cp1)
        backend.save(cp2)

        completed = backend.get_list_by_flow_and_run(flow_id, wf_run_id, status=CheckpointStatus.COMPLETED)
        assert len(completed) == 1
        assert completed[0].status == CheckpointStatus.COMPLETED

    def test_get_list_by_flow_and_run_with_limit(self, backend: CheckpointBackend):
        """get_list_by_flow_and_run respects limit."""
        flow_id = "limit-flow"
        wf_run_id = "limit-wf-run"
        for i in range(5):
            cp = FlowCheckpoint(flow_id=flow_id, run_id=f"run-{i}", wf_run_id=wf_run_id)
            backend.save(cp)
            time.sleep(0.01)

        results = backend.get_list_by_flow_and_run(flow_id, wf_run_id, limit=2)
        assert len(results) == 2

    def test_get_list_by_flow_and_run_empty(self, backend: CheckpointBackend):
        """get_list_by_flow_and_run returns empty list when no match."""
        results = backend.get_list_by_flow_and_run("nonexistent-flow", "nonexistent-run")
        assert results == []

    def test_get_latest_by_flow_and_run(self, backend: CheckpointBackend):
        """get_latest_by_flow_and_run returns the most recent checkpoint."""
        flow_id = "latest-flow"
        wf_run_id = "latest-wf-run"
        cp1 = FlowCheckpoint(flow_id=flow_id, run_id="run-1", wf_run_id=wf_run_id)
        backend.save(cp1)
        time.sleep(0.01)
        cp2 = FlowCheckpoint(flow_id=flow_id, run_id="run-2", wf_run_id=wf_run_id)
        backend.save(cp2)

        latest = backend.get_latest_by_flow_and_run(flow_id, wf_run_id)
        assert latest is not None
        assert latest.run_id == "run-2"

    def test_get_latest_by_flow_and_run_empty(self, backend: CheckpointBackend):
        """get_latest_by_flow_and_run returns None when no match."""
        result = backend.get_latest_by_flow_and_run("nonexistent-flow", "nonexistent-run")
        assert result is None


class TestInMemoryBackend(BackendTestMixin):
    """Tests for InMemory checkpoint backend."""

    @pytest.fixture
    def backend(self) -> InMemory:
        """Create an in-memory backend for testing."""
        return InMemory()

    def test_clear(self, backend: InMemory):
        """Test clearing all checkpoints."""
        for i in range(5):
            checkpoint = FlowCheckpoint(flow_id="test-flow", run_id=f"run-{i}")
            backend.save(checkpoint)

        assert len(backend._checkpoints) == 5

        backend.clear()

        assert len(backend._checkpoints) == 0

    def test_get_list_by_run(self, backend: InMemory):
        """Test listing checkpoints by run_id."""
        run_id = "test-run-123"

        cp1 = FlowCheckpoint(flow_id="flow-1", run_id=run_id)
        cp2 = FlowCheckpoint(flow_id="flow-2", run_id=run_id)
        cp3 = FlowCheckpoint(flow_id="flow-3", run_id="other-run")

        backend.save(cp1)
        backend.save(cp2)
        backend.save(cp3)

        checkpoints = backend.get_list_by_run(run_id)
        assert len(checkpoints) == 2
        assert all(cp.run_id == run_id for cp in checkpoints)

    def test_isolation_between_saves(self, backend: InMemory):
        """Test that saved checkpoints are isolated from modifications."""
        checkpoint = FlowCheckpoint(flow_id="test-flow", run_id="run-1")
        backend.save(checkpoint)

        checkpoint.status = CheckpointStatus.FAILED

        loaded = backend.load(checkpoint.id)
        assert loaded.status == CheckpointStatus.ACTIVE

    def test_thread_safety(self, backend: InMemory):
        """Test thread-safe operations."""
        import threading

        errors = []

        def save_checkpoint(i: int):
            try:
                checkpoint = FlowCheckpoint(flow_id="thread-test", run_id=f"run-{i}")
                backend.save(checkpoint)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=save_checkpoint, args=(i,)) for i in range(100)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert len(backend._checkpoints) == 100


class TestFileSystemBackend(BackendTestMixin):
    """Tests for FileSystem backend."""

    @pytest.fixture
    def backend(self, tmp_path: Path) -> FileSystem:
        """Create a filesystem backend in a temporary directory."""
        return FileSystem(base_path=str(tmp_path / ".dynamiq" / "checkpoints"))

    def test_creates_base_directory(self, tmp_path: Path):
        base_dir = tmp_path / ".dynamiq" / "checkpoints"
        backend = FileSystem(base_path=str(base_dir))
        assert backend._base_dir.exists()

    def test_save_creates_flow_and_run_directories(self, backend: FileSystem):
        checkpoint = FlowCheckpoint(flow_id="test-flow", run_id="run-1")
        backend.save(checkpoint)

        flow_dir = backend._flow_dir("test-flow")
        assert flow_dir.exists()

        run_dirs = [d for d in flow_dir.iterdir() if d.is_dir()]
        assert len(run_dirs) == 1
        assert "run-1" in run_dirs[0].name

    def test_checkpoint_file_has_timestamp_and_id(self, backend: FileSystem):
        checkpoint = FlowCheckpoint(flow_id="test-flow", run_id="run-1")
        backend.save(checkpoint)

        flow_dir = backend._flow_dir("test-flow")
        run_dir = next(d for d in flow_dir.iterdir() if d.is_dir())
        cp_files = list(run_dir.glob("*.json"))
        assert len(cp_files) == 1
        assert checkpoint.id.replace("-", "_") in cp_files[0].stem or checkpoint.id[:8] in cp_files[0].stem

    def test_sanitizes_flow_id_in_directory_name(self, backend: FileSystem):
        checkpoint = FlowCheckpoint(flow_id="test/flow:name", run_id="run-1")
        backend.save(checkpoint)

        flow_dir = backend._flow_dir("test/flow:name")
        assert flow_dir.exists()
        assert "/" not in flow_dir.name
        assert ":" not in flow_dir.name

    def test_get_list_by_run(self, backend: FileSystem):
        run_id = "test-run-123"
        cp1 = FlowCheckpoint(flow_id="flow-1", run_id=run_id)
        cp2 = FlowCheckpoint(flow_id="flow-2", run_id=run_id)
        cp3 = FlowCheckpoint(flow_id="flow-3", run_id="other-run")
        backend.save(cp1)
        backend.save(cp2)
        backend.save(cp3)

        checkpoints = backend.get_list_by_run(run_id)
        assert len(checkpoints) == 2

    def test_delete_removes_file(self, backend: FileSystem):
        flow_id = "test-flow"
        cp1 = FlowCheckpoint(flow_id=flow_id, run_id="run-1")
        cp2 = FlowCheckpoint(flow_id=flow_id, run_id="run-2")
        backend.save(cp1)
        backend.save(cp2)

        assert len(backend.get_list_by_flow(flow_id)) == 2

        backend.delete(cp1.id)

        checkpoints = backend.get_list_by_flow(flow_id)
        assert len(checkpoints) == 1
        assert checkpoints[0].id == cp2.id

    def test_multiple_checkpoints_per_run_in_same_directory(self, backend: FileSystem):
        flow_id = "test-flow"
        run_id = "run-1"
        cp1 = FlowCheckpoint(flow_id=flow_id, run_id=run_id)
        cp2 = FlowCheckpoint(flow_id=flow_id, run_id=run_id)
        backend.save(cp1)
        backend.save(cp2)

        flow_dir = backend._flow_dir(flow_id)
        run_dirs = [d for d in flow_dir.iterdir() if d.is_dir()]
        assert len(run_dirs) == 1

        cp_files = list(run_dirs[0].glob("*.json"))
        assert len(cp_files) == 2

    def test_same_wf_run_id_reuses_directory(self, backend: FileSystem):
        """Checkpoints with the same wf_run_id but different run_ids land in one directory."""
        flow_id = "reuse-dir-flow"
        wf_run_id = "shared-wf-run"

        cp1 = FlowCheckpoint(flow_id=flow_id, run_id="run-1", wf_run_id=wf_run_id)
        cp2 = FlowCheckpoint(flow_id=flow_id, run_id="run-2", wf_run_id=wf_run_id)
        cp3 = FlowCheckpoint(flow_id=flow_id, run_id="run-3", wf_run_id=wf_run_id)

        backend.save(cp1)
        backend.save(cp2)
        backend.save(cp3)

        flow_dir = backend._flow_dir(flow_id)
        run_dirs = [d for d in flow_dir.iterdir() if d.is_dir()]
        assert len(run_dirs) == 1
        assert run_dirs[0].name.endswith(f"__{wf_run_id}")

        cp_files = list(run_dirs[0].glob("*.json"))
        assert len(cp_files) == 3

    def test_different_wf_run_ids_get_separate_directories(self, backend: FileSystem):
        """Checkpoints with different wf_run_ids create separate directories."""
        flow_id = "separate-dir-flow"

        cp1 = FlowCheckpoint(flow_id=flow_id, run_id="run-1", wf_run_id="wf-run-a")
        cp2 = FlowCheckpoint(flow_id=flow_id, run_id="run-2", wf_run_id="wf-run-b")

        backend.save(cp1)
        backend.save(cp2)

        flow_dir = backend._flow_dir(flow_id)
        run_dirs = sorted(d for d in flow_dir.iterdir() if d.is_dir())
        assert len(run_dirs) == 2
        assert run_dirs[0].name.endswith("__wf-run-a") or run_dirs[1].name.endswith("__wf-run-a")
        assert run_dirs[0].name.endswith("__wf-run-b") or run_dirs[1].name.endswith("__wf-run-b")

    def test_directory_uses_wf_run_id_over_run_id(self, backend: FileSystem):
        """Run directory name uses wf_run_id when available, not run_id."""
        flow_id = "wf-pref-flow"
        cp = FlowCheckpoint(flow_id=flow_id, run_id="flow-run-id", wf_run_id="wf-run-id")
        backend.save(cp)

        flow_dir = backend._flow_dir(flow_id)
        run_dir = next(d for d in flow_dir.iterdir() if d.is_dir())
        assert run_dir.name.endswith("__wf-run-id")
        assert "flow-run-id" not in run_dir.name

    def test_directory_falls_back_to_run_id_without_wf_run_id(self, backend: FileSystem):
        """Run directory name uses run_id when wf_run_id is None."""
        flow_id = "fallback-flow"
        cp = FlowCheckpoint(flow_id=flow_id, run_id="my-run-id", wf_run_id=None)
        backend.save(cp)

        flow_dir = backend._flow_dir(flow_id)
        run_dir = next(d for d in flow_dir.iterdir() if d.is_dir())
        assert run_dir.name.endswith("__my-run-id")

    def test_get_list_by_run_uses_suffix_match(self, backend: FileSystem):
        """get_list_by_run uses suffix match, not substring."""
        cp1 = FlowCheckpoint(flow_id="flow-1", run_id="run-123")
        cp2 = FlowCheckpoint(flow_id="flow-2", run_id="run-1234")
        backend.save(cp1)
        backend.save(cp2)

        results = backend.get_list_by_run("run-123")
        assert len(results) == 1
        assert results[0].run_id == "run-123"

    def test_get_list_by_flow_and_run_filesystem(self, backend: FileSystem):
        """get_list_by_flow_and_run scopes to a specific flow's run directory."""
        flow_id = "scoped-flow"
        wf_run_id = "scoped-wf-run"

        cp1 = FlowCheckpoint(flow_id=flow_id, run_id="run-1", wf_run_id=wf_run_id)
        cp2 = FlowCheckpoint(flow_id=flow_id, run_id="run-2", wf_run_id=wf_run_id)
        cp3 = FlowCheckpoint(flow_id="other-flow", run_id="run-3", wf_run_id=wf_run_id)

        backend.save(cp1)
        backend.save(cp2)
        backend.save(cp3)

        results = backend.get_list_by_flow_and_run(flow_id, wf_run_id)
        assert len(results) == 2
        assert all(r.flow_id == flow_id for r in results)


class TestAsyncBackendMethods:
    """Tests for async backend methods."""

    @pytest.fixture
    def backend(self) -> InMemory:
        return InMemory()

    @pytest.mark.asyncio
    async def test_async_save_and_load(self, backend: InMemory):
        """Test async save and load (delegates to sync)."""
        checkpoint = FlowCheckpoint(flow_id="test-flow", run_id="run-1")

        checkpoint_id = await backend.save_async(checkpoint)
        loaded = await backend.load_async(checkpoint_id)

        assert loaded is not None
        assert loaded.id == checkpoint.id

    @pytest.mark.asyncio
    async def test_async_update(self, backend: InMemory):
        """Test async update (delegates to sync)."""
        checkpoint = FlowCheckpoint(flow_id="test-flow", run_id="run-1")
        await backend.save_async(checkpoint)

        checkpoint.status = CheckpointStatus.COMPLETED
        await backend.update_async(checkpoint)

        loaded = await backend.load_async(checkpoint.id)
        assert loaded.status == CheckpointStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_async_delete(self, backend: InMemory):
        """Test async delete (delegates to sync)."""
        checkpoint = FlowCheckpoint(flow_id="test-flow", run_id="run-1")
        await backend.save_async(checkpoint)

        deleted = await backend.delete_async(checkpoint.id)
        assert deleted is True

        loaded = await backend.load_async(checkpoint.id)
        assert loaded is None
