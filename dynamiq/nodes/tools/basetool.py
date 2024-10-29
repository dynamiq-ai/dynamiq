from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, Field

from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ActionParsingException
from dynamiq.runnables import RunnableConfig


class DefaultInputSchema(BaseModel):
    input: str


class BaseTool(BaseModel, ABC):
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "Base Tool"
    description: str = "Base tool description"
    input_schema: type[BaseModel] = Field(default=DefaultInputSchema)

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
        try:
            return self.run_tool(self.input_schema(**input_data), config=config, **kwargs)
        except Exception as e:
            raise ActionParsingException(f"Error: Invalid format for input data: {e}")
