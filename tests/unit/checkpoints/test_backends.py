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

    def test_creates_directory_structure(self, tmp_path: Path):
        """Test that backend creates necessary directories."""
        base_dir = tmp_path / ".dynamiq" / "checkpoints"
        backend = FileSystem(base_path=str(base_dir))

        assert backend._data_dir.exists()
        assert backend._flow_index_dir.exists()

    def test_checkpoint_files_exist(self, backend: FileSystem):
        """Test that checkpoint files are created."""
        checkpoint = FlowCheckpoint(flow_id="test-flow", run_id="run-1")
        backend.save(checkpoint)

        checkpoint_file = backend._data_dir / f"{checkpoint.id}.json"
        assert checkpoint_file.exists()

    def test_index_file_created(self, backend: FileSystem):
        """Test that flow index file is created."""
        checkpoint = FlowCheckpoint(flow_id="test-flow", run_id="run-1")
        backend.save(checkpoint)

        index_file = backend._flow_index_dir / "test-flow.json"
        assert index_file.exists()

    def test_sanitizes_flow_id_for_index(self, backend: FileSystem):
        """Test that special characters in flow_id are sanitized."""
        checkpoint = FlowCheckpoint(flow_id="test/flow:name", run_id="run-1")
        backend.save(checkpoint)

        index_file = backend._flow_index_dir / "test_flow_name.json"
        assert index_file.exists()

    def test_get_list_by_run(self, backend: FileSystem):
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

    def test_delete_updates_index(self, backend: FileSystem):
        """Test that deleting a checkpoint updates the flow index."""
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
