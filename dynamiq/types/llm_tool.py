from pydantic import BaseModel


class ToolFunctionParameters(BaseModel):
    type: str
    properties: dict[str, dict]
    required: list[str]


class ToolFunction(BaseModel):
    name: str
    description: str
    parameters: ToolFunctionParameters


class Tool(BaseModel):
    type: str = "function"
    function: ToolFunction
