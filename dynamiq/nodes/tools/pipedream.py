from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, create_model

from dynamiq.connections import PipedreamOAuth2 as PipedreamConnection
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.nodes.tools.mcp import rename_keys_recursive
from dynamiq.nodes.tools.utils import create_file_from_url
from dynamiq.runnables import RunnableConfig
from dynamiq.types.cancellation import check_cancellation
from dynamiq.utils.logger import logger

FIELD_TYPES_MAPPING = {
    "string": str,
    "string[]": list[str],
    "integer": int,
    "integer[]": list[int],
    "number": float,
    "number[]": list[float],
    "boolean": bool,
    "boolean[]": list[bool],
    "object": dict[str, Any],
    "object[]": list[dict[str, Any]],
    "any": Any,
    "$.interface.http": Any,
    "$.interface.timer": Any,
    "$.service.db": Any,
    "http_request": Any,
    "data_store": Any,
    "app": Any,
}
SUCCESS_CODES = [200]
RECOVERABLE_CODES = [400, 401, 402, 422]


# TODO: Wrap config fields (like id, external_user_id, etc.) inside a "config" BaseModel structure(can be passed
#  as dict) to avoid overriding top-level parameter names in the node structure.
class Pipedream(ConnectionNode):
    """
    A tool for executing workflows and automations using the Pipedream API.

    Attributes:
        name (str): Name of the tool
        description (str): Description of the tool
        group (Literal[NodeGroup.TOOLS]): The group the node belongs to
        connection (HttpApiKey): The Pipedream API connection
        input_props(dict[str, Any]): Schema of the input parameters
        timeout (float): Request timeout in seconds
        dynamic_props_id (str): Specified identification for dynamic configured additional properties
        stash_id (str): Specified identification for file stash, used for actions with files
    """

    name: str = "pipedream"
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    connection: PipedreamConnection
    timeout: float = 180
    input_schema: type[BaseModel]
    input_props: dict[str, Any]
    description: str = ""
    external_user_id: str
    action_id: str
    configurable_props: dict[str, Any]
    dynamic_props_id: str | None = None
    stash_id: str | bool | None = None

    def __init__(self, input_props: dict[str, Any], configurable_props: dict[str, Any], **kwargs):
        input_props = rename_keys_recursive(input_props, {"type": "type_"})
        input_schema = self.get_input_schema(input_props, configurable_props=configurable_props)
        super().__init__(
            input_schema=input_schema,
            input_props=input_props,
            configurable_props=configurable_props,
            **kwargs,
        )
        self.description = self._generate_description()

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

        desc: list[str] = [self.description if self.description else f"{self.name} tool"]

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

        configured_fields: list[str] = [
            name
            for name, field in self.configurable_props.items()
            if not (isinstance(field, dict) and field.get("authProvisionId") is not None)
        ]

        if configured_fields:
            desc.append("\nAlready configured parameters, that can be overridden:")
            for field_name in sorted(configured_fields):
                field = self.configurable_props[field_name]
                desc.append(f"- {field_name}: {field}")

        return "\n".join(desc)

    def get_input_schema(self, schema_dict, configurable_props) -> type[BaseModel]:
        """
        Creates an input schema based on provided JSON schema.

        Args:
            schema_dict (dict[str, Any]): A JSON schema dictionary describing the tool's expected input.
        """
        schema_dict = rename_keys_recursive(schema_dict, {"type_": "type"})
        fields = {}
        if isinstance(schema_dict, dict):
            config_props = schema_dict.get("configurable_props", []) or schema_dict.get("configurableProps", [])
        elif isinstance(schema_dict, list):
            config_props = schema_dict
        else:
            config_props = []

        for prop in config_props:
            name = prop["name"]
            description = prop.get("description", "")
            is_optional = prop.get("optional", False)
            raw_type = prop.get("type", "string")
            field_type = FIELD_TYPES_MAPPING.get(raw_type, Any)
            if configurable_props.get(name):
                is_optional = True
            if raw_type not in ("app", "alert"):
                if is_optional:
                    fields[name] = (Optional[field_type], Field(default=prop.get("default"), description=description))
                else:
                    fields[name] = (field_type, Field(..., description=description))

        ConfiguredPropsModel = create_model("ConfiguredProps", **fields)

        return ConfiguredPropsModel

    def _build_request(self, input_data: BaseModel) -> tuple[str, dict]:
        base_url = self.connection.url or "https://api.pipedream.com/v1/"
        url = f"{base_url}/connect/{self.connection.project_id}/actions/run"
        payload = {
            "external_user_id": self.external_user_id,
            "id": self.action_id,
            "configured_props": {
                **self.configurable_props,
                **{k: v for k, v in input_data.model_dump().items() if v is not None},
            },
            **({"stash_id": self.stash_id} if self.stash_id is not None else {}),
            **({"dynamic_props_id": self.dynamic_props_id} if self.dynamic_props_id else {}),
        }
        return url, payload

    def _check_response_status(self, response: Any) -> None:
        if response.status_code not in SUCCESS_CODES:
            error_message = f"Pipedream API request failed with status code: {response.status_code}"
            logger.error(f"Tool {self.name} - {error_message}")
            recoverable = response.status_code in RECOVERABLE_CODES
            raise ToolExecutionException(f"{error_message} and response: {response.text}", recoverable=recoverable)
        if (
            status := response.json().get("exports", {}).get("debug", {}).get("status")
        ) and status not in SUCCESS_CODES:
            error_message = f"Pipedream API request failed with status code: {status}"
            logger.error(f"Tool {self.name} - {error_message}")
            recoverable = response.status_code in RECOVERABLE_CODES
            raise ToolExecutionException(
                f"{error_message} and response: {response.json().get('exports', {}).get('debug', {}).get('data')}",
                recoverable=recoverable,
            )

    def _process_response_files(self, response_json: dict) -> list:
        files = []
        if file_uploads := response_json.get("exports", {}).get("$filestash_uploads", None):
            for file_upload in file_uploads:
                file_name = file_upload.get("path")
                file_url = file_upload.get("get_url")
                if file_url:
                    files.append(create_file_from_url(file_url, file_name, self.timeout))
        return files

    def _build_output(self, response: Any, files: list) -> dict[str, Any]:
        if self.is_optimized_for_agents:
            content = response.text
        else:
            content = response.json()
        return {"content": content, "files": files}

    def execute(self, input_data: BaseModel, config: RunnableConfig = None, **kwargs):
        """Execute the specific workflow logic.

        Args:
            input_data (PipedreamInputSchema): The input data containing email details
            config (RunnableConfig, optional): Configuration for the execution
            **kwargs: Additional keyword arguments

        Returns:
            dict: A dictionary containing:
                - content (dict): The API response content
                - status_code (int): The HTTP status code

        Raises:
            ToolExecutionException: If the API request fails or required parameters are missing
            ValueError: If neither text nor html content is provided
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

            try:
                files = self._process_response_files(response.json())
            except Exception as e:
                logger.error(f"Tool {self.name} - Unexpected error during file processing: {str(e)}")
                raise ToolExecutionException(f"Unexpected error during file processing:  {str(e)}", recoverable=True)

            return self._build_output(response, files)
        except ToolExecutionException:
            raise
        except Exception as e:
            logger.error(f"Tool {self.name} - Unexpected error during execution: {str(e)}")
            raise ToolExecutionException(f"Unexpected error during execution:  {str(e)}", recoverable=False)

    async def execute_async(self, input_data: BaseModel, config: RunnableConfig = None, **kwargs):
        """Native async execution path mirroring ``execute``.

        The HTTP call is awaited natively. File-attachment download still uses the sync
        ``create_file_from_url`` helper, so the file-processing block is offloaded to a
        thread via ``asyncio.to_thread`` to keep the event loop unblocked. Adding async
        file downloads is out of scope for this PR.
        """
        import asyncio

        logger.debug(f"Tool {self.name} - Starting execution with input data: {input_data.model_dump()}")

        config = ensure_config(config)
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

            try:
                files = await asyncio.to_thread(self._process_response_files, response.json())
            except Exception as e:
                logger.error(f"Tool {self.name} - Unexpected error during file processing: {str(e)}")
                raise ToolExecutionException(f"Unexpected error during file processing:  {str(e)}", recoverable=True)

            return self._build_output(response, files)
        except ToolExecutionException:
            raise
        except Exception as e:
            logger.error(f"Tool {self.name} - Unexpected error during execution: {str(e)}")
            raise ToolExecutionException(f"Unexpected error during execution:  {str(e)}", recoverable=False)
