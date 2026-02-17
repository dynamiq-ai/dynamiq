"""Unit tests for checkpoint serialization."""

import base64
import json
from datetime import date, datetime
from io import BytesIO
from uuid import uuid4

from dynamiq.checkpoints.checkpoint import CheckpointStatus, FlowCheckpoint, NodeCheckpointState, PendingInputContext
from dynamiq.utils import decode_reversible, encode_reversible


class TestEncodeReversible:
    """Tests for encode_reversible function."""

    def test_encode_datetime(self):
        dt = datetime(2024, 1, 15, 10, 30, 45)
        result = encode_reversible(dt)
        assert result == {"__datetime__": "2024-01-15T10:30:45"}

    def test_encode_date(self):
        d = date(2024, 1, 15)
        result = encode_reversible(d)
        assert result == {"__date__": "2024-01-15"}

    def test_encode_uuid(self):
        test_uuid = uuid4()
        result = encode_reversible(test_uuid)
        assert result == {"__uuid__": str(test_uuid)}

    def test_encode_enum(self):
        result = encode_reversible(CheckpointStatus.ACTIVE)
        assert result == "active"

        result = encode_reversible(CheckpointStatus.COMPLETED)
        assert result == "completed"

    def test_encode_pydantic_model(self):
        checkpoint = FlowCheckpoint(flow_id="test-flow", run_id="run-1")
        result = encode_reversible(checkpoint)

        assert isinstance(result, dict)
        assert result["flow_id"] == "test-flow"
        assert result["run_id"] == "run-1"

    def test_encode_set(self):
        test_set = {1, 2, 3}
        result = encode_reversible(test_set)
        assert "__set__" in result
        assert sorted(result["__set__"]) == [1, 2, 3]

    def test_encode_bytes(self):
        test_bytes = b"Hello, World!"
        result = encode_reversible(test_bytes)

        assert "__bytes__" in result
        assert isinstance(result["__bytes__"], str)

    def test_encode_bytesio(self):
        bio = BytesIO(b"File content here")
        bio.name = "test_file.pdf"
        result = encode_reversible(bio)

        assert "__bytesio__" in result
        assert isinstance(result["__bytesio__"], str)
        assert result["name"] == "test_file.pdf"

    def test_encode_bytesio_without_name(self):
        bio = BytesIO(b"Anonymous content")
        result = encode_reversible(bio)

        assert "__bytesio__" in result
        assert result["name"] is None

    def test_encode_object_with_dict(self):
        class CustomObject:
            def __init__(self):
                self.field1 = "value1"
                self.field2 = 42

        obj = CustomObject()
        result = encode_reversible(obj)

        assert result == {"field1": "value1", "field2": 42}

    def test_encode_unsupported_type_returns_value(self):
        class UnsupportedType:
            __slots__ = ()

        obj = UnsupportedType()
        result = encode_reversible(obj)
        assert result is obj


class TestDecodeReversible:
    """Tests for decode_reversible function."""

    def test_decode_bytes(self):
        original = b"Hello, World!"
        encoded = {"__bytes__": base64.b64encode(original).decode("utf-8")}

        result = decode_reversible(encoded)
        assert result == original

    def test_decode_bytesio(self):
        original_content = b"File content for testing"
        encoded = {
            "__bytesio__": base64.b64encode(original_content).decode("utf-8"),
            "name": "document.pdf",
        }

        result = decode_reversible(encoded)

        assert isinstance(result, BytesIO)
        assert result.getvalue() == original_content
        assert result.name == "document.pdf"

    def test_decode_bytesio_without_name(self):
        original_content = b"Anonymous file content"
        encoded = {
            "__bytesio__": base64.b64encode(original_content).decode("utf-8"),
            "name": None,
        }

        result = decode_reversible(encoded)

        assert isinstance(result, BytesIO)
        assert result.getvalue() == original_content
        assert not hasattr(result, "name") or result.name is None

    def test_decode_datetime(self):
        encoded = {"__datetime__": "2024-01-15T10:30:45"}
        result = decode_reversible(encoded)
        assert result == datetime(2024, 1, 15, 10, 30, 45)

    def test_decode_date(self):
        encoded = {"__date__": "2024-01-15"}
        result = decode_reversible(encoded)
        assert result == date(2024, 1, 15)

    def test_decode_uuid(self):
        from uuid import UUID

        test_uuid_str = "12345678-1234-5678-1234-567812345678"
        encoded = {"__uuid__": test_uuid_str}
        result = decode_reversible(encoded)
        assert result == UUID(test_uuid_str)

    def test_decode_set(self):
        encoded = {"__set__": [1, 2, 3]}
        result = decode_reversible(encoded)
        assert result == {1, 2, 3}

    def test_decode_regular_dict(self):
        test_dict = {"key": "value", "number": 42}
        result = decode_reversible(test_dict)
        assert result == test_dict


class TestFlowCheckpointToJson:
    """Tests for FlowCheckpoint.to_json() method."""

    def test_serialize_simple_checkpoint(self):
        checkpoint = FlowCheckpoint(flow_id="test-flow", run_id="run-123")

        result = checkpoint.to_json()

        assert isinstance(result, str)
        data = json.loads(result)
        assert data["flow_id"] == "test-flow"
        assert data["run_id"] == "run-123"

    def test_serialize_checkpoint_with_node_states(self):
        checkpoint = FlowCheckpoint(flow_id="test-flow", run_id="run-1")

        node_state = NodeCheckpointState(
            node_id="node-1",
            node_type="TestNode",
            status="success",
            output_data={"result": 42},
            internal_state={"key": "value"},
        )
        checkpoint.node_states["node-1"] = node_state
        checkpoint.completed_node_ids.append("node-1")

        result = checkpoint.to_json()
        data = json.loads(result)

        assert "node-1" in data["node_states"]
        assert data["node_states"]["node-1"]["status"] == "success"
        assert data["completed_node_ids"] == ["node-1"]

    def test_serialize_checkpoint_with_pending_input(self):
        checkpoint = FlowCheckpoint(
            flow_id="test-flow",
            run_id="run-1",
            status=CheckpointStatus.PENDING_INPUT,
            pending_inputs={
                "agent-1": PendingInputContext(
                    node_id="agent-1",
                    prompt="What should I do?",
                    timestamp=datetime.utcnow(),
                    metadata={"tool": "human_feedback"},
                )
            },
        )

        result = checkpoint.to_json()
        data = json.loads(result)

        assert data["status"] == "pending_input"
        assert data["pending_inputs"]["agent-1"]["node_id"] == "agent-1"
        assert data["pending_inputs"]["agent-1"]["prompt"] == "What should I do?"

    def test_serialize_with_complex_input_data(self):
        checkpoint = FlowCheckpoint(
            flow_id="test-flow",
            run_id="run-1",
            original_input={
                "query": "test query",
                "nested": {"key": "value", "list": [1, 2, 3]},
                "unicode": "日本語テスト",
            },
        )

        result = checkpoint.to_json()
        data = json.loads(result)

        assert data["original_input"]["unicode"] == "日本語テスト"
        assert data["original_input"]["nested"]["list"] == [1, 2, 3]


class TestFlowCheckpointFromJson:
    """Tests for FlowCheckpoint.from_json() method."""

    def test_deserialize_simple_checkpoint(self):
        json_str = json.dumps(
            {
                "id": "cp-123",
                "flow_id": "test-flow",
                "run_id": "run-1",
                "status": "active",
                "node_states": {},
                "completed_node_ids": [],
                "pending_node_ids": [],
                "original_input": None,
                "original_config": None,
                "pending_inputs": {},
                "created_at": "2024-01-15T10:30:45",
                "updated_at": "2024-01-15T10:30:45",
                "version": "1.0",
                "dynamiq_version": None,
                "metadata": {},
                "parent_checkpoint_id": None,
            }
        )

        checkpoint = FlowCheckpoint.from_json(json_str)

        assert checkpoint.id == "cp-123"
        assert checkpoint.flow_id == "test-flow"
        assert checkpoint.status == CheckpointStatus.ACTIVE

    def test_deserialize_preserves_node_states(self):
        json_str = json.dumps(
            {
                "id": "cp-123",
                "flow_id": "test-flow",
                "run_id": "run-1",
                "status": "completed",
                "node_states": {
                    "node-1": {
                        "node_id": "node-1",
                        "node_type": "TestNode",
                        "status": "success",
                        "output_data": {"result": 42},
                    }
                },
                "completed_node_ids": ["node-1"],
                "pending_node_ids": [],
                "original_input": {"query": "test"},
                "original_config": None,
                "pending_inputs": {},
                "created_at": "2024-01-15T10:30:45",
                "updated_at": "2024-01-15T10:30:45",
                "version": "1.0",
                "dynamiq_version": None,
                "metadata": {},
                "parent_checkpoint_id": None,
            }
        )

        checkpoint = FlowCheckpoint.from_json(json_str)

        assert "node-1" in checkpoint.node_states
        assert checkpoint.node_states["node-1"].status == "success"
        assert checkpoint.node_states["node-1"].output_data == {"result": 42}


class TestRoundTripSerialization:
    """Tests for round-trip serialization (to_json -> from_json)."""

    def test_roundtrip_simple(self):
        original = FlowCheckpoint(flow_id="test-flow", run_id="run-1")

        serialized = original.to_json()
        restored = FlowCheckpoint.from_json(serialized)

        assert restored.id == original.id
        assert restored.flow_id == original.flow_id
        assert restored.run_id == original.run_id
        assert restored.status == original.status

    def test_roundtrip_bytesio_in_input_data(self):
        file_content = b"PDF content or image bytes"
        bio = BytesIO(file_content)
        bio.name = "document.pdf"

        original = FlowCheckpoint(
            flow_id="test-flow",
            run_id="run-1",
            original_input={"file": bio, "query": "analyze this"},
        )

        serialized = original.to_json()
        restored = FlowCheckpoint.from_json(serialized)

        assert isinstance(restored.original_input["file"], BytesIO)
        assert restored.original_input["file"].getvalue() == file_content
        assert restored.original_input["file"].name == "document.pdf"
        assert restored.original_input["query"] == "analyze this"

    def test_roundtrip_bytes_in_node_output(self):
        original = FlowCheckpoint(flow_id="test-flow", run_id="run-1")

        node_state = NodeCheckpointState(
            node_id="node-1",
            node_type="FileProcessor",
            status="success",
            output_data={"processed": b"binary result data"},
        )
        original.mark_node_complete("node-1", node_state)

        serialized = original.to_json()
        restored = FlowCheckpoint.from_json(serialized)

        assert restored.node_states["node-1"].output_data["processed"] == b"binary result data"

    def test_roundtrip_with_all_fields(self):
        original = FlowCheckpoint(
            flow_id="test-flow",
            run_id="run-1",
            workflow_id="wf-1",
            status=CheckpointStatus.COMPLETED,
            original_input={"query": "test", "nested": {"a": 1}},
            original_config={"timeout": 30},
            metadata={"custom": "data"},
        )

        node_state = NodeCheckpointState(
            node_id="node-1",
            node_type="Agent",
            status="success",
            input_data={"input": "data"},
            output_data={"output": "result"},
            internal_state={"history": ["msg1", "msg2"]},
        )
        original.mark_node_complete("node-1", node_state)

        serialized = original.to_json()
        restored = FlowCheckpoint.from_json(serialized)

        assert restored.workflow_id == original.workflow_id
        assert restored.original_input == original.original_input
        assert restored.original_config == original.original_config
        assert "node-1" in restored.node_states
        assert restored.node_states["node-1"].internal_state == {"history": ["msg1", "msg2"]}

    def test_roundtrip_with_pending_input(self):
        original = FlowCheckpoint(flow_id="test-flow", run_id="run-1")

        original.mark_pending_input("agent-1", "What should I do?", {"tool": "human"})

        serialized = original.to_json()
        restored = FlowCheckpoint.from_json(serialized)

        assert restored.status == CheckpointStatus.PENDING_INPUT
        assert "agent-1" in restored.pending_inputs
        assert restored.pending_inputs["agent-1"].node_id == "agent-1"
        assert restored.pending_inputs["agent-1"].prompt == "What should I do?"

    def test_roundtrip_with_multiple_pending_inputs(self):
        """Test parallel HITL with multiple pending inputs."""
        original = FlowCheckpoint(flow_id="test-flow", run_id="run-1")

        original.mark_pending_input("agent-1", "Approve action A?", {"tool": "human"})
        original.mark_pending_input("agent-2", "Approve action B?", {"tool": "human"})

        serialized = original.to_json()
        restored = FlowCheckpoint.from_json(serialized)

        assert restored.status == CheckpointStatus.PENDING_INPUT
        assert len(restored.pending_inputs) == 2
        assert "agent-1" in restored.pending_inputs
        assert "agent-2" in restored.pending_inputs
        assert restored.pending_inputs["agent-1"].prompt == "Approve action A?"
        assert restored.pending_inputs["agent-2"].prompt == "Approve action B?"


class TestByteSerialization:
    """Tests for bytes serialization methods."""

    def test_to_bytes(self):
        checkpoint = FlowCheckpoint(flow_id="test-flow", run_id="run-1")

        result = checkpoint.to_bytes()

        assert isinstance(result, bytes)
        assert b"test-flow" in result

    def test_from_bytes(self):
        checkpoint = FlowCheckpoint(flow_id="test-flow", run_id="run-1")

        serialized = checkpoint.to_bytes()
        restored = FlowCheckpoint.from_bytes(serialized)

        assert restored.flow_id == checkpoint.flow_id
        assert restored.run_id == checkpoint.run_id

    def test_bytes_roundtrip(self):
        original = FlowCheckpoint(
            flow_id="test-flow",
            run_id="run-1",
            original_input={"query": "test"},
            metadata={"key": "value"},
        )

        serialized = original.to_bytes()
        restored = FlowCheckpoint.from_bytes(serialized)

        assert restored.id == original.id
        assert restored.original_input == original.original_input
        assert restored.metadata == original.metadata


class TestJsonIntegration:
    """Tests for JSON integration with standard json module."""

    def test_json_dumps_with_encoder(self):
        checkpoint = FlowCheckpoint(flow_id="test-flow", run_id="run-1")

        result = json.dumps(checkpoint.model_dump(), default=encode_reversible)

        assert isinstance(result, str)
        assert "test-flow" in result

    def test_json_loads_with_decoder(self):
        json_str = json.dumps(
            {
                "data": {"__bytes__": base64.b64encode(b"test").decode()},
                "normal": "value",
            }
        )

        result = json.loads(json_str, object_hook=decode_reversible)

        assert result["data"] == b"test"
        assert result["normal"] == "value"
