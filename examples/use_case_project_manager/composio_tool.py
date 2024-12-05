from typing import Literal, Optional

from composio import Action, ComposioToolSet
from pydantic import BaseModel, ConfigDict, Field

from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import Node, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger


def generate_pydantic_model(schema_dict: dict, model_name: str = "GeneratedModel") -> type[BaseModel]:
    """
    Generates a Pydantic BaseModel class from a schema dictionary.

    Args:
        schema_dict (dict): Dictionary containing field definitions
        model_name (str): Name for the generated model class

    Returns:
        type[BaseModel]: Generated Pydantic model class
    """
    annotations = {}
    field_definitions = {}

    type_mapping = {"string": str, "integer": int, "number": float, "boolean": bool, "array": list}

    for field_name, field_info in schema_dict.items():
        field_type = type_mapping.get(field_info.get("type", "string"), str)

        # Handle arrays
        if field_type == list and "items" in field_info:
            item_type = type_mapping.get(field_info["items"].get("type", "string"), str)
            field_type = list[item_type]

        # Create Field with description and default if present
        field_kwargs = {
            "description": field_info.get("description", ""),
        }

        if "default" in field_info:
            field_kwargs["default"] = field_info["default"]
            annotations[field_name] = Optional[field_type]
        else:
            annotations[field_name] = field_type

        field_definitions[field_name] = Field(**field_kwargs)

    # Create the namespace for the class
    namespace = {"__annotations__": annotations, **field_definitions}

    # Create the model class
    return type(model_name, (BaseModel,), namespace)


class ComposioTool(Node):
    """
    A tool for interacting with Composio API.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "Composio Tool"
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    action: Action
    description: str = ""
    api_key: str
    _input_schema: type[BaseModel] | None = None

    def __init__(self, **data):
        super().__init__(**data)
        self._setup_schema()
        self.description = self._generate_description()

    def _setup_schema(self) -> None:
        """Set up the input schema based on the action."""
        composio_toolset = ComposioToolSet(api_key=self.api_key)
        action_schema = composio_toolset.get_action_schemas(actions=[self.action])[0]
        logger.info(f"Tool {self.name} - Setting up input schema for action: {action_schema}")
        self._input_schema = generate_pydantic_model(
            action_schema.parameters.properties, f"ComposioInput_{self.action.name}"
        )
        self.name = f"Composio Tool - {action_schema.display_name}"

    @property
    def input_schema(self) -> type[BaseModel]:
        """Get the input schema for the action."""
        if self._input_schema is None:
            raise ValueError("Input schema not initialized")
        return self._input_schema

    def _generate_description(self) -> str:
        """Generate a detailed description of the tool based on the action schema."""
        desc = [
            f"Tool for {self.name}\n",
            "Required Parameters:",
        ]

        schema_fields = self.input_schema.model_fields

        required_fields = [name for name, field in schema_fields.items() if field.is_required() is not False]

        if required_fields:
            for field_name in sorted(required_fields):
                field = schema_fields[field_name]
                desc.append(f"- {field_name}: {field.description}")
        else:
            desc.append("None")

        desc.append("Optional Parameters:")

        optional_fields = [name for name, field in schema_fields.items() if field.is_required() is False]

        if optional_fields:
            for field_name in sorted(optional_fields):
                field = schema_fields[field_name]
                desc.append(f"- {field_name}: {field.description}")
        else:
            desc.append("None")

        return "\n".join(desc)

    def execute(self, input_data: BaseModel, config: RunnableConfig = None, **kwargs):
        """Execute the Composio action."""
        logger.debug(f"Tool {self.name} - Starting execution with action: {self.action}")

        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        try:
            toolset = ComposioToolSet()

            params = {k: v for k, v in input_data.model_dump().items() if v is not None}

            logger.debug(f"Tool {self.name} - Executing action with params: {params}")

            result = toolset.execute_action(action=self.action, params=params)
            logger.info(f"Tool {self.name} - Execution result: {result}")
            if result.get("successfull", False):
                return {"content": result.get("data")}
            else:
                raise ToolExecutionException(f"Failed to execute {self.name}: {result.get('error')}", recoverable=True)

        except Exception as e:
            logger.error(f"Tool {self.name} - Error during execution: {str(e)}")
            raise ToolExecutionException(f"Failed to execute action {self.name}: {str(e)}", recoverable=True)
