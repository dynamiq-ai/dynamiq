import json
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.nodes.node import Node, ensure_config
from dynamiq.nodes.types import NodeGroup
from dynamiq.runnables import RunnableConfig
from dynamiq.utils import JsonWorkflowEncoder


class AnyToJSONTransformerInputSchema(BaseModel):
    value: Any = Field(..., description="Parameter to provide value to transform.")


class AnyToJSON(Node):
    group: Literal[NodeGroup.TRANSFORMERS] = NodeGroup.TRANSFORMERS
    name: str = "AnyToJSON"
    description: str = "Node that transforms value to JSON"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_schema: ClassVar[type[AnyToJSONTransformerInputSchema]] = AnyToJSONTransformerInputSchema

    def execute(
        self, input_data: AnyToJSONTransformerInputSchema, config: RunnableConfig = None, **kwargs
    ) -> dict[str, Any]:
        """
        Transform value to JSON.

        Args:
            input_data (AnyToJSONTransformerInputSchema): input data for the tool, which includes
                value to transform.
            config (RunnableConfig, optional): Configuration for the runnable, including callbacks.
            **kwargs: Additional arguments passed to the execution context.

        Returns:
            dict[str, Any]: A dictionary containing the JSON.
        """
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        value = input_data.value
        try:
            if isinstance(value, BaseModel):
                result = value.model_dump_json()
            else:
                result = json.dumps(value, cls=JsonWorkflowEncoder)
            return {"content": result}
        except (json.decoder.JSONDecodeError, TypeError):
            raise ValueError(f"Invalid data for transformation: {value}")


class JSONToAnyTransformerInputSchema(BaseModel):
    value: str = Field(..., description="Parameter to provide JSON to transform.")


class JSONToAny(Node):
    group: Literal[NodeGroup.TRANSFORMERS] = NodeGroup.TRANSFORMERS
    name: str = "JSONToAny"
    description: str = "Node that transforms JSON to an object"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_schema: ClassVar[type[JSONToAnyTransformerInputSchema]] = JSONToAnyTransformerInputSchema

    def execute(
        self, input_data: JSONToAnyTransformerInputSchema, config: RunnableConfig = None, **kwargs
    ) -> dict[str, Any]:
        """
        Transform JSON to a Python object.

        Args:
            input_data (JSONToAnyTransformerInputSchema): input data for the tool, which includes
                string value to transform.
            config (RunnableConfig, optional): Configuration for the runnable, including callbacks.
            **kwargs: Additional arguments passed to the execution context.

        Returns:
            dict[str, Any]: A dictionary containing data from JSON as a Python object.
        """
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        value = input_data.value
        try:
            result = json.loads(value)
            return {"content": result}
        except json.decoder.JSONDecodeError:
            raise ValueError(f"Invalid JSON: {value}")
