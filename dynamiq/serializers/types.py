from pydantic import BaseModel, ConfigDict

from dynamiq import Workflow
from dynamiq.connections import BaseConnection
from dynamiq.flows import Flow
from dynamiq.nodes import Node


class WorkflowYamlData(BaseModel):
    """Data model for the Workflow YAML."""

    connections: dict[str, BaseConnection]
    nodes: dict[str, Node]
    flows: dict[str, Flow]
    workflows: dict[str, Workflow]

    model_config = ConfigDict(arbitrary_types_allowed=True)
