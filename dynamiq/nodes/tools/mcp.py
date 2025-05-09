import asyncio
import importlib.util
import inspect
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

from datamodel_code_generator import DataModelType, InputFileType, generate
from mcp import ClientSession
from pydantic import BaseModel, ConfigDict, PrivateAttr

from dynamiq.connections import MCPSse, MCPStdio, MCPStreamableHTTP
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils import is_called_from_async_context
from dynamiq.utils.logger import logger


def rename_keys_recursive(data: dict[str, Any] | list[str], key_map: dict[str, str]) -> Any:
    """
    Recursively renames keys in a nested dictionary based on a provided key mapping.
    """
    if isinstance(data, dict):
        return {key_map.get(key, key): rename_keys_recursive(value, key_map) for key, value in data.items()}
    elif isinstance(data, list):
        return [rename_keys_recursive(item, key_map) for item in data]
    return data


class ServerMetadata(BaseModel):
    id: str | None = None
    name: str | None = None
    description: str | None = None

    model_config = ConfigDict(extra="allow")


class MCPTool(ConnectionNode):
    """
    A tool that interacts with the MCP server, enabling execution of specific server-side functions.

    Attributes:
      group (Literal[NodeGroup.TOOLS]): Node group.
      name (str): Node name.
      description (str): Node description.
      input_schema (ClassVar[type[BaseModel]]): The schema that defines the expected structure of tool's input.
      connection (MCPSse | MCPStdio | MCPStreamableHTTP): Connection module for the MCP server.
      server_metadata (ServerMetadata): Server metadata for tracing.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str
    description: str
    input_schema: type[BaseModel]
    json_input_schema: dict[str, Any]
    connection: MCPSse | MCPStdio | MCPStreamableHTTP
    server_metadata: ServerMetadata = field(default_factory=dict)

    def __init__(self, json_input_schema: dict[str, Any], **kwargs):
        """
        Initializes the MCP tool with a given input schema and additional parameters.

        Args:
            input_schema (dict[str, Any]): JSON schema to define input fields.
            **kwargs
        """
        input_schema = MCPTool.get_input_schema(json_input_schema)
        json_input_schema = rename_keys_recursive(json_input_schema, {"type": "type_"})
        super().__init__(input_schema=input_schema, json_input_schema=json_input_schema, **kwargs)

    @property
    def to_dict_exclude_params(self):
        parent_dict = super().to_dict_exclude_params.copy()
        parent_dict.update(
            {
                "input_schema": True,
            }
        )
        return parent_dict

    @staticmethod
    def get_input_schema(schema_dict) -> type[BaseModel]:
        """
        Creates an input schema based on provided MCP schema.

        Args:
            schema_dict (dict[str, Any]): A JSON schema dictionary describing the tool's expected input.
        """
        schema_dict = rename_keys_recursive(schema_dict, {"type_": "type"})

        for _, props in schema_dict.get("properties", {}).items():
            enum_values = props.pop("enum", None)
            if enum_values:
                description = props.get("description", "")
                enum_description = f" Allowed values: {', '.join(map(str, enum_values))}."
                props["description"] = description.rstrip() + enum_description

        input_schema = json.dumps(schema_dict)

        with TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "model.py"

            generate(
                input_schema,
                input_file_type=InputFileType.JsonSchema,
                output=out_path,
                output_model_type=DataModelType.PydanticV2BaseModel,
            )

            spec = importlib.util.spec_from_file_location("dynamiq.nodes.tools.MCPTool", out_path)
            generated_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(generated_module)
            generated_classes = [
                cls
                for name, cls in inspect.getmembers(generated_module, inspect.isclass)
                if cls.__module__ == generated_module.__name__
            ]
            return generated_classes[0]

    def execute(self, input_data: BaseModel, config: RunnableConfig | None = None, **kwargs) -> dict[str, Any]:
        """
        Executes the MCP tool synchronously with the provided input.

        Args:
            input_data (BaseModel): Input data for the tool execution.
            config (RunnableConfig, optional): Configuration for the runnable, including callbacks.
            **kwargs: Additional arguments passed to the execution context.

        Returns:
            dict[str, Any]: A dictionary containing the tool's output.
        """
        return asyncio.run(self.execute_async(input_data, config, **kwargs))

    async def execute_async(
        self, input_data: BaseModel, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Executes the MCP tool asynchronously.

        Args:
            input_data (BaseModel): Input data for the tool execution.

        Returns:
            dict[str, Any]: A dictionary containing the tool's output.
        """
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        input_dict = input_data.model_dump()

        try:
            async with self.connection.connect() as result:
                read, write = result[:2]
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(self.name, input_dict)
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to get results. Error: {str(e)}")
            raise ToolExecutionException(
                f"Tool '{self.name}' failed to call tool from the MCP server."
                f"Error: {str(e)}. Please analyze the error and take appropriate action.",
                recoverable=True,
            )

        logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")
        return {"content": result}


class MCPServer(ConnectionNode):
    """
    A tool that manages connections to MCP servers and initializes MCP tools.

    Attributes:
      group (Literal[NodeGroup.TOOLS]): Node group.
      name (str): Node name.
      description (str): Node description.
      connection (MCPSse | MCPStdio | MCPStreamableHTTP): Connection module for the MCP server.
      include_tools (list[str]): Names of tools to include. If empty, all tools are included.
      exclude_tools (list[str]): Names of tools to exclude.
      _mcp_tools (dict[str, MCPTool]): Internal dict of initialized MCP tools.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "MCP Tool"
    description: str = "The tool used to initialize available MCP tools based on provided server parameters."
    connection: MCPSse | MCPStdio | MCPStreamableHTTP

    include_tools: list[str] = field(default_factory=list)
    exclude_tools: list[str] = field(default_factory=list)
    _mcp_tools: dict[str, MCPTool] = PrivateAttr(default_factory=dict)

    async def initialize_tools(self):
        """
        Initializes the MCP tool list from the client session.

        Returns:
            None
        """
        async with self.connection.connect() as result:
            read, write = result[:2]
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                for tool in tools.tools:
                    self._mcp_tools[tool.name] = MCPTool(
                        name=tool.name,
                        description=tool.description or "MCP Tool",
                        json_input_schema=tool.inputSchema,
                        connection=self.connection,
                        server_metadata=ServerMetadata(id=self.id, name=self.name, description=self.description),
                    )

        logger.info(f"Tool {self.name}: {len(self._mcp_tools)} MCP tools initialized from a server.")

    def get_mcp_tools(self, select_all: bool = False) -> list[MCPTool]:
        """
        Synchronously fetches and initializes MCP tools if not already available.

        Args:
            select_all (bool): If True, returns all tools regardless of filtering.

        Returns:
            list[MCPTool]: A list of initialized MCPTool instances.
        """
        if is_called_from_async_context():
            with ThreadPoolExecutor() as executor:
                future = executor.submit(lambda: asyncio.run(self.get_mcp_tools_async(select_all=select_all)))
                return future.result()
        return asyncio.run(self.get_mcp_tools_async(select_all=select_all))

    async def get_mcp_tools_async(self, select_all: bool = False) -> list[MCPTool]:
        """
        Asynchronously fetches and initializes MCP tools if not already available.

        Args:
            select_all (bool): If True, returns all tools regardless of filtering.

        Returns:
            list[MCPTool]: A list of initialized MCPTool instances.
        """
        if not self._mcp_tools:
            await self.initialize_tools()

        if select_all or not self.include_tools and not self.exclude_tools:
            return list(self._mcp_tools.values())

        if self.include_tools:
            return [v for k, v in self._mcp_tools.items() if k in self.include_tools and k not in self.exclude_tools]

        return [v for k, v in self._mcp_tools.items() if k not in self.exclude_tools]

    def execute(self, **kwargs):
        """
        Disabled for the MCP server. Use `get_mcp_tools()` to access individual tools.

        Raises:
            NotImplementedError: Always, because this method is not supported.
        """
        raise NotImplementedError("Use `get_mcp_tools()` to access individual tool instances.")
