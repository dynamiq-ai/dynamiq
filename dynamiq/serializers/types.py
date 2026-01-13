from pydantic import BaseModel, ConfigDict, Field, ValidationError

from dynamiq import Workflow
from dynamiq.connections import BaseConnection
from dynamiq.flows import Flow
from dynamiq.nodes import Node


class RequirementData(BaseModel):
    """Information about a dict/object that requires external resolution.

    This model tracks any dict in YAML that has both $type and $id fields,
    which indicates it needs to be resolved via an external API before workflow initialization.
    """

    type: str = Field(..., alias="$type", description="The $type field value")
    id: str = Field(..., alias="$id", description="The $id field - unique identifier for external resolution")

    @classmethod
    def from_dict(cls, data: dict) -> "RequirementData | None":
        """Create RequirementData from dict if it has $type and $id keys."""
        try:
            return cls.model_validate(data)
        except ValidationError:
            return None


class WorkflowYamlData(BaseModel):
    """Data model for the Workflow YAML."""

    connections: dict[str, BaseConnection]
    nodes: dict[str, Node]
    flows: dict[str, Flow]
    workflows: dict[str, Workflow]

    model_config = ConfigDict(arbitrary_types_allowed=True)
