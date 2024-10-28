from dynamiq.nodes.node import ConnectionNode
from typing import Literal, Type, Any
from dynamiq.nodes import NodeGroup
from pydantic import BaseModel, field_validator, ValidationInfo, Field
from dynamiq.utils.logger import logger
from dynamiq.runnables import RunnableConfig
from abc import ABC, abstractmethod

class DefaultInputSchema(BaseModel, ABC):
    input: str

class Tool(ConnectionNode):
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "base_tool"
    description: str = "base tool"
    input_shema: Type[BaseModel] = Field(default=DefaultInputSchema)

    @field_validator('input_shema')
    def passwords_match(cls, v: BaseModel, info: ValidationInfo) -> BaseModel:
        # Check if input_shema is correct
        return v
    
    @abstractmethod
    def run_tool(self):
        pass

    @property
    def input_params(self) -> str:
        """Return list of input parameters along with their type and description."""
        params = []
        for name, field in self.input_shema.model_fields.items():
            tags = {}
            if field.json_schema_extra:
                tags = field.json_schema_extra
            description = field.description or "No description"
            field_type: str = field.annotation.__name__
            match field_type:
                case "str":
                    field_type = "string"

            params.append((name, field_type, description, tags))
        return params

    def execute(self, input_data: dict[str, Any], config: RunnableConfig | None = None, **kwargs) -> dict[str, Any]:
        """Executes the requested action based on the input data."""
        return self.run_tool(self.input_shema(**input_data), config, **kwargs)
