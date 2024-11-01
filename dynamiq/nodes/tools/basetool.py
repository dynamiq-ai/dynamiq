from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel

from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ActionParsingException
from dynamiq.runnables import RunnableConfig


class DefaultInputSchema(BaseModel):
    input: str


class BaseTool(BaseModel, ABC):
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "Base Tool"
    description: str = "Base tool description"
    _input_schema: type[BaseModel] = DefaultInputSchema

    @property
    def input_schema(self) -> type[BaseModel]:
        return self._input_schema

    @classmethod
    def get_input_schema(cls) -> type[BaseModel]:
        return cls._input_schema

    @abstractmethod
    def run_tool(self) -> dict[str, Any]:
        pass

    @property
    def input_params(self) -> list[tuple]:
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
