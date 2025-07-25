import os.path
from datetime import date, datetime
from enum import Enum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import Field

from dynamiq import Workflow
from dynamiq.connections import BaseConnection
from dynamiq.connections.managers import get_connection_manager
from dynamiq.flows import Flow
from dynamiq.nodes import Node, NodeGroup
from dynamiq.runnables import RunnableConfig
from dynamiq.serializers.loaders.yaml import WorkflowYAMLLoader


class ProcessingMode(str, Enum):
    FAST = "fast"
    ACCURATE = "accurate"
    BALANCED = "balanced"


class ConnectionType(str, Enum):
    HTTP = "http"
    WEBSOCKET = "websocket"
    TCP = "tcp"


class AdvancedHTTPConnection(BaseConnection):
    url: str = "https://api.example.com"
    api_key: str = "test-api-key"
    timeout: int = 30

    connection_type: ConnectionType = ConnectionType.HTTP
    created_at: datetime
    valid_until: date
    session_uuid: UUID = Field(default_factory=uuid4)
    retry_delays: list[float] = Field(default=[1.0, 2.5, 5.0])
    metadata: dict[str, Any] = Field(default_factory=lambda: {"version": "1.0", "region": "us-east"})

    @property
    def type(self) -> str:
        """Return the correct type for YAML serialization."""
        return "tests.integration.workflows.test_serialization.AdvancedHTTPConnection"

    def connect(self):
        return {
            "url": self.url,
            "api_key": self.api_key,
            "timeout": self.timeout,
            "connection_type": self.connection_type.value,
            "session_uuid": str(self.session_uuid),
        }


class AdvancedProcessorNode(Node):
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "Advanced Processor"
    description: str = "An advanced node with complex data types"

    processing_mode: ProcessingMode = ProcessingMode.BALANCED
    created_at: datetime
    scheduled_date: date
    node_uuid: UUID = Field(default_factory=uuid4)
    weights: list[float] = Field(default=[0.1, 0.5, 0.4])
    config_map: dict[str, Any] = Field(default_factory=lambda: {"max_retries": 3, "use_cache": True})

    connection: AdvancedHTTPConnection | None = None

    @property
    def type(self) -> str:
        """Return the correct type for YAML serialization."""
        return "tests.integration.workflows.test_serialization.AdvancedProcessorNode"

    def execute(self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        """Execute the advanced processor."""
        return {
            "output": "text",
            "processed": True,
            "processing_mode": self.processing_mode.value,
            "node_uuid": str(self.node_uuid),
            "weights_sum": sum(self.weights),
        }


def test_workflow_yaml_serialization_and_deserialization(tmp_path):
    now = datetime.now()

    original_conn = AdvancedHTTPConnection(
        id="roundtrip-conn",
        url="https://roundtrip.service.com",
        api_key="roundtrip-key-789",
        connection_type=ConnectionType.TCP,
        created_at=now,
        valid_until=now.date(),
        session_uuid=uuid4(),
        retry_delays=[1.0, 2.0, 4.0],
        metadata={"env": "test", "debug": True},
    )

    original_node = AdvancedProcessorNode(
        id="roundtrip-processor",
        name="Roundtrip Processor",
        processing_mode=ProcessingMode.FAST,
        created_at=now,
        scheduled_date=now.date(),
        node_uuid=uuid4(),
        connection=original_conn,
    )

    original_flow = Flow(id="roundtrip-flow", name="Roundtrip Flow", nodes=[original_node])
    original_workflow = Workflow(id="roundtrip-workflow", flow=original_flow, version="1.5.0")

    yaml_file = os.path.join(tmp_path, "workflow.yaml")
    original_workflow.to_yaml_file(yaml_file)

    with get_connection_manager() as cm:
        wf_data = WorkflowYAMLLoader.load(file_path=yaml_file, connection_manager=cm, init_components=True)
        loaded_workflow = Workflow.from_yaml_file_data(file_data=wf_data)

    assert loaded_workflow.id == original_workflow.id
    assert loaded_workflow.name == original_workflow.name
    assert loaded_workflow.version == original_workflow.version

    original_flow_data = original_workflow.flow
    loaded_flow_data = loaded_workflow.flow
    assert loaded_flow_data.id == original_flow_data.id
    assert loaded_flow_data.name == original_flow_data.name
    assert len(loaded_flow_data.nodes) == len(original_flow_data.nodes)

    original_node_data = original_workflow.flow.nodes[0]
    loaded_node_data = loaded_workflow.flow.nodes[0]

    assert loaded_node_data.id == original_node_data.id
    assert loaded_node_data.name == original_node_data.name
    assert loaded_node_data.processing_mode == original_node_data.processing_mode
    assert loaded_node_data.created_at == original_node_data.created_at
    assert loaded_node_data.scheduled_date == original_node_data.scheduled_date
    assert loaded_node_data.node_uuid == original_node_data.node_uuid
    assert loaded_node_data.weights == original_node_data.weights
    assert loaded_node_data.config_map == original_node_data.config_map

    original_conn_data = original_node_data.connection
    loaded_conn_data = loaded_node_data.connection

    assert loaded_conn_data.id == original_conn_data.id
    assert loaded_conn_data.url == original_conn_data.url
    assert loaded_conn_data.api_key == original_conn_data.api_key
    assert loaded_conn_data.connection_type == original_conn_data.connection_type
    assert loaded_conn_data.created_at == original_conn_data.created_at
    assert loaded_conn_data.valid_until == original_conn_data.valid_until
    assert loaded_conn_data.session_uuid == original_conn_data.session_uuid
    assert loaded_conn_data.retry_delays == original_conn_data.retry_delays
    assert loaded_conn_data.metadata == original_conn_data.metadata
