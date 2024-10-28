from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator

from dynamiq.nodes import NodeGroup
from dynamiq.runnables import RunnableConfig


class DefaultInputSchema(BaseModel, ABC):
    input: str


class ToolMixin:
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "base_tool"
    description: str = "base tool"
    input_schema: type[BaseModel] = Field(default=DefaultInputSchema)

    @field_validator("input_schema")
    def passwords_match(cls, v: BaseModel, info: ValidationInfo) -> BaseModel:
        # Check if input_schema is correct
        return v

    @abstractmethod
    def run_tool(self):
        pass

    @property
    def input_params(self) -> str:
        """Return list of input parameters along with their type and description."""
        params = []
        for name, field in self.input_schema.model_fields.items():
            tags = {}
            if field.json_schema_extra:
                tags = field.json_schema_extra
            description = field.description or "No description"
            field_type: str = field.annotation.__name__

            params.append((name, field_type, description, tags))
        return params

    def execute(self, input_data: dict[str, Any], config: RunnableConfig | None = None, **kwargs) -> dict[str, Any]:
        """Executes the requested action based on the input data."""
        return self.run_tool(self.input_schema(**input_data), config=config, **kwargs)
