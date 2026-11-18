"""Host-authorized tools whose opaque execution context is invisible to the model."""

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from dynamiq.connections import Http
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode
from dynamiq.nodes.tools.mcp import create_input_schema_from_json_schema
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.remote_control import post_control_json, validate_control_connection


class RemoteContextTool(ConnectionNode):
    """Call a configured endpoint with typed tool input and separate host-owned control context."""

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    input_schema: type[BaseModel]
    connection: Http
    parameters: dict[str, Any]
    request_timeout_seconds: float = Field(default=10, gt=0, le=60)

    def __init__(self, parameters: dict[str, Any], **kwargs):
        schema = create_input_schema_from_json_schema(parameters, "RemoteToolInput")
        super().__init__(parameters=parameters, input_schema=schema, **kwargs)

    @model_validator(mode="after")
    def validate_endpoint(self):
        validate_control_connection(self.connection)
        if self.parameters.get("type") != "object":
            raise ValueError("Remote tool parameters must describe a JSON object")
        return self

    @property
    def to_dict_exclude_params(self) -> dict:
        return super().to_dict_exclude_params | {"input_schema": True}

    def execute(self, input_data: BaseModel, config: RunnableConfig | None = None, **kwargs) -> dict:
        try:
            trusted = config.trusted_context if config else {}
            control = trusted.get("control_context")
            if not isinstance(control, dict):
                raise ValueError("Remote tool control context is unavailable")
            output = post_control_json(
                self.client,
                self.connection,
                {
                    "run_id": config.run_id,
                    "app_id": trusted.get("app_id"),
                    "deployment_id": trusted.get("deployment_id"),
                    "control_context": control,
                    "input": input_data.model_dump(mode="json"),
                },
                self.request_timeout_seconds,
                1024 * 1024,
            )
        except Exception:
            raise ToolExecutionException("Remote context tool request failed", recoverable=False) from None
        return {"content": output}
