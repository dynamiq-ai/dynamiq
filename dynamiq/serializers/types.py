from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from dynamiq import Workflow
from dynamiq.connections import BaseConnection
from dynamiq.flows import Flow
from dynamiq.nodes import Node


class ObjectType(str, Enum):
    """Enum for object types in YAML data that require special handling."""

    REQUIREMENT = "requirement"


class RequirementData(BaseModel):
    """Information about a dict/object that requires external resolution via requirement_id.

    This model tracks any dict in YAML that has object='requirement', which indicates
    it needs to be resolved via an external API before workflow initialization.
    """

    object: str = Field(default=ObjectType.REQUIREMENT.value, description="Object type identifier")
    requirement_id: str = Field(..., description="Unique identifier for external resolution")

    @classmethod
    def from_dict(cls, data: dict) -> "RequirementData | None":
        """Extract RequirementData from dict if it has object='requirement' and requirement_id."""
        if data.get("object") != ObjectType.REQUIREMENT.value:
            return None

        requirement_id = data.get("requirement_id")
        if not requirement_id:
            return None

        return cls(requirement_id=requirement_id)


class WorkflowYamlData(BaseModel):
    """Data model for the Workflow YAML."""

    connections: dict[str, BaseConnection]
    nodes: dict[str, Node]
    flows: dict[str, Flow]
    workflows: dict[str, Workflow]

    model_config = ConfigDict(arbitrary_types_allowed=True)
