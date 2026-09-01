from typing import Any, Literal

from pydantic import BaseModel, PrivateAttr

from dynamiq.connections import Composio as ComposioConnection
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.nodes.tools.mcp import (
    create_input_schema_from_json_schema,
    get_json_schema_definitions,
    rename_keys_recursive,
    resolve_root_object_schema,
)
from dynamiq.runnables import RunnableConfig
from dynamiq.types.cancellation import check_cancellation
from dynamiq.utils.logger import logger

DEFAULT_COMPOSIO_URL = "https://backend.composio.dev/api/v3"
SUCCESS_CODES = [200]
RECOVERABLE_CODES = [400, 401, 402, 422, 429]


class Composio(ConnectionNode):
    """
    A tool for executing a single Composio action (tool) on behalf of a Composio user.

    Attributes:
        name (str): Name of the tool
        description (str): Description of the tool
        group (Literal[NodeGroup.TOOLS]): The group the node belongs to
        connection (ComposioConnection): The Composio API connection
        timeout (float): Request timeout in seconds
        input_props (dict[str, Any]): The tool's `input_parameters` JSON Schema
        user_id (str): The Composio user id the action runs as
        toolkit_slug (str): Slug of the Composio toolkit (app), e.g. `gmail`
        tool_slug (str): Slug of the Composio tool (action), e.g. `GMAIL_SEND_EMAIL`
        tool_version (str | None): Pins the tool version the arguments were configured against
        connected_account_id (str | None): Pins the action to a specific connected account
        arguments (dict[str, Any]): Argument values configured at design time
    """

    name: str = "composio"
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    connection: ComposioConnection
    timeout: float = 180
    input_schema: type[BaseModel]
    input_props: dict[str, Any] = {}
    description: str = ""
    user_id: str
    toolkit_slug: str
    tool_slug: str
    tool_version: str | None = None
    connected_account_id: str | None = None
    arguments: dict[str, Any] = {}

    # The description Composio supplied for the action, before `_generate_description` appends the
    # parameter listing onto it. `None` means it was never captured, which only happens if the node
    # was built without running `__init__`.
    _source_description: str | None = PrivateAttr(default=None)

    def __init__(self, input_props: dict[str, Any] | None = None, **kwargs):
        arguments = kwargs.get("arguments") or {}
        # A tool that declares no parameters is persisted with an empty or absent schema - the node
        # type serializes `input_props` with `omitempty`, and Composio itself reports "no declared
        # schema" as either `{}` or `null` - so every such spelling has to yield an empty schema
        # rather than a construction error.
        if not isinstance(input_props, dict):
            input_props = {}
        input_props = rename_keys_recursive(input_props, {"type": "type_"})
        input_schema = self.get_input_schema(input_props, arguments=arguments)
        super().__init__(
            input_schema=input_schema,
            input_props=input_props,
            **kwargs,
        )
        self._source_description = self.description
        self.description = self._generate_description()

    def to_dict(self, **kwargs) -> dict:
        data = super().to_dict(**kwargs)
        # At runtime `description` carries the parameter listing that `_generate_description` appended
        # to the description Composio supplied. Persisting that listing would feed it back in on the
        # next load, where it is appended again, so it would compound on every save/load cycle.
        if self._source_description is not None and "description" in data:
            data["description"] = self._source_description
        return data

    @property
    def to_dict_exclude_params(self):
        parent_dict = super().to_dict_exclude_params.copy()
        parent_dict.update(
            {
                "input_schema": True,
            }
        )
        return parent_dict

    def _generate_description(self) -> str:
        """
        Generates a detailed description of the tool based on the input schema.

        Returns:
            str: A formatted description of the tool and its capabilities
        """
        schema_fields: dict[str, Any] = self.input_schema.model_fields
        logger.debug(f"Tool {self.name} - Generating description from schema fields")

        # The platform supplies a description snapshotted from Composio when the action was picked.
        # Constructed directly, there is none, so fall back to the slugs rather than the node name:
        # an agent picking between tools needs to know which action this is.
        fallback = f"Executes the Composio tool {self.tool_slug} from the {self.toolkit_slug} toolkit."
        desc: list[str] = [self.description or fallback]

        required_fields: list[str] = [name for name, field in schema_fields.items() if field.is_required() is not False]
        if required_fields:
            desc.append("\nRequired Parameters:")
            for field_name in sorted(required_fields):
                field = schema_fields[field_name]
                desc.append(f"- {field_name} ({str(field.annotation)}): {field.description}")

        optional_fields: list[str] = [name for name, field in schema_fields.items() if field.is_required() is False]
        if optional_fields:
            desc.append("\nOptional Parameters:")
            for field_name in sorted(optional_fields):
                field = schema_fields[field_name]
                desc.append(f"- {field_name} ({str(field.annotation)}): {field.description}")

        if self.arguments:
            desc.append("\nAlready configured parameters, that can be overridden:")
            for field_name in sorted(self.arguments):
                desc.append(f"- {field_name}: {self.arguments[field_name]}")

        return "\n".join(desc)

    def get_input_schema(self, schema_dict: dict[str, Any], arguments: dict[str, Any]) -> type[BaseModel]:
        """
        Creates an input schema based on the tool's `input_parameters` JSON Schema.

        Args:
            schema_dict (dict[str, Any]): A JSON schema dictionary describing the tool's expected input.
            arguments (dict[str, Any]): Argument values already configured at design time.
        """
        if not isinstance(schema_dict, dict):
            schema_dict = {}
        schema_dict = rename_keys_recursive(schema_dict, {"type_": "type"})

        # A root composed purely of $ref/allOf is resolved to the object schema it stands for before
        # anything below inspects it. The schema builder resolves the same way, so rewriting the
        # unresolved root instead would edit keys it goes on to discard. The definitions live on the
        # original root and have to be carried across, since the resolved schema no longer holds them.
        definitions = get_json_schema_definitions(schema_dict)
        schema_dict = resolve_root_object_schema(schema_dict, definitions)

        # Composio marks optionality the Pydantic way, so only `required` is authoritative. A parameter
        # already configured at design time is dropped from it: the runtime input may override the
        # configured value, but does not have to supply one.
        required = schema_dict.get("required") or []
        schema_dict = {**schema_dict, "required": [name for name in required if name not in arguments]}

        properties = schema_dict.get("properties")
        if arguments and isinstance(properties, dict):
            # A configured parameter falls back to its configured value, so it must not keep the schema
            # default as well: the model emits that default whenever the caller leaves the parameter
            # out, and it wins over `arguments` when the request payload is assembled.
            schema_dict = {**schema_dict, "properties": dict(properties)}
            for name in arguments:
                prop = properties.get(name)
                if isinstance(prop, dict) and "default" in prop:
                    schema_dict["properties"][name] = {key: value for key, value in prop.items() if key != "default"}

        return create_input_schema_from_json_schema(schema_dict, "ComposioToolSchema", definitions=definitions)

    def _build_request(self, input_data: BaseModel) -> tuple[str, dict]:
        base_url = (self.connection.url or DEFAULT_COMPOSIO_URL).rstrip("/")
        url = f"{base_url}/tools/execute/{self.tool_slug}"
        payload = {
            "arguments": {
                **self.arguments,
                # by_alias restores the original JSON Schema property names for parameters whose
                # names are not valid Python identifiers.
                **{k: v for k, v in input_data.model_dump(by_alias=True).items() if v is not None},
            },
            "user_id": self.user_id,
            # Omitting the version lets Composio resolve its own default, which could differ
            # from the schema the arguments were configured against.
            **({"version": self.tool_version} if self.tool_version else {}),
            **({"connected_account_id": self.connected_account_id} if self.connected_account_id else {}),
        }
        return url, payload

    def _check_response_status(self, response: Any) -> None:
        if response.status_code not in SUCCESS_CODES:
            error_message = f"Composio API request failed with status code: {response.status_code}"
            logger.error(f"Tool {self.name} - {error_message}")
            recoverable = response.status_code in RECOVERABLE_CODES
            raise ToolExecutionException(f"{error_message} and response: {response.text}", recoverable=recoverable)

    def _check_execution_result(self, response_json: dict) -> None:
        """Fail on a tool-level error.

        Composio answers 200 even when the action itself failed, so `successful` in the body -
        not the HTTP status - carries the real outcome. The failure is reported as unrecoverable
        because an execute must never be replayed: it may have side effects.
        """
        if not response_json.get("successful", True):
            error_message = f"Composio tool {self.tool_slug} execution failed"
            logger.error(f"Tool {self.name} - {error_message}")
            raise ToolExecutionException(f"{error_message} with error: {response_json.get('error')}", recoverable=False)

    def _build_output(self, response: Any) -> dict[str, Any]:
        if self.is_optimized_for_agents:
            return {"content": response.text}
        return {"content": response.json().get("data")}

    def execute(self, input_data: BaseModel, config: RunnableConfig = None, **kwargs):
        """Execute the Composio action.

        Args:
            input_data (BaseModel): The action arguments, validated against the tool's input schema
            config (RunnableConfig, optional): Configuration for the execution
            **kwargs: Additional keyword arguments

        Returns:
            dict: A dictionary containing:
                - content (Any): The `data` payload of the Composio response

        Raises:
            ToolExecutionException: If the API request fails or the action itself reports a failure
        """
        logger.debug(f"Tool {self.name} - Starting execution with input data: {input_data.model_dump()}")

        config = ensure_config(config)
        check_cancellation(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        try:
            url, payload = self._build_request(input_data)
            response = self.client.request(
                method="POST",
                url=url,
                headers=self.connection.conn_params,
                json=payload,
                timeout=self.timeout,
            )
            self._check_response_status(response)
            self._check_execution_result(response.json())

            return self._build_output(response)
        except ToolExecutionException:
            raise
        except Exception as e:
            logger.error(f"Tool {self.name} - Unexpected error during execution: {str(e)}")
            raise ToolExecutionException(f"Unexpected error during execution:  {str(e)}", recoverable=False)

    async def execute_async(self, input_data: BaseModel, config: RunnableConfig = None, **kwargs):
        """Native async execution path mirroring ``execute``."""
        logger.debug(f"Tool {self.name} - Starting execution with input data: {input_data.model_dump()}")

        config = ensure_config(config)
        check_cancellation(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        try:
            url, payload = self._build_request(input_data)
            client = await self.get_async_client()
            response = await client.request(
                method="POST",
                url=url,
                headers=self.connection.conn_params,
                json=payload,
                timeout=self.timeout,
            )
            self._check_response_status(response)
            self._check_execution_result(response.json())

            return self._build_output(response)
        except ToolExecutionException:
            raise
        except Exception as e:
            logger.error(f"Tool {self.name} - Unexpected error during execution: {str(e)}")
            raise ToolExecutionException(f"Unexpected error during execution:  {str(e)}", recoverable=False)
