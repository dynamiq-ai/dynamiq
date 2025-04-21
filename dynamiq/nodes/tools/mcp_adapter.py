import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, Field, create_model

from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils import is_called_from_async_context
from dynamiq.utils.logger import logger


class SseServerParameters(BaseModel):
    """
    Parameters for configuring a Server-Sent Events client connection.
    """

    url: str = Field(..., description="The SSE endpoint URL to connect to.")
    headers: dict[str, Any] | None = Field(default=None, description="Optional headers to include in the SSE request.")
    timeout: float = Field(default=5.0, description="Timeout in seconds for establishing the initial connection.")
    sse_read_timeout: float = Field(
        default=60 * 5, description="Timeout in seconds for reading SSE messages. Defaults to 5 minutes."
    )


ServerParameters = StdioServerParameters | SseServerParameters


async def get_client(server_params: ServerParameters):
    """
    Creates an asynchronous client context manager based on server parameters.

    Args:
        server_params (ServerParameters): Parameters specifying the connection type and settings.

    Returns:
        Async context manager for a client connection (either stdio or SSE).
    """
    if isinstance(server_params, StdioServerParameters):
        return stdio_client(server_params)
    elif isinstance(server_params, SseServerParameters):
        return sse_client(
            url=server_params.url,
            headers=server_params.headers,
            timeout=server_params.timeout,
            sse_read_timeout=server_params.sse_read_timeout,
        )
    raise TypeError("Unsupported server parameter type.")


class MCPTool(Node):
    """
    A tool that interacts with the MCP server, enabling execution of specific server-side functions.

    Attributes:
      group (Literal[NodeGroup.TOOLS]): Node group.
      name (str): Node name.
      description (str): Node description.
      input_schema (ClassVar[type[BaseModel]]): The schema that defines the expected structure of tool's input.
      server_params (ServerParameters): The parameters for connecting to the MCP server.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str
    description: str
    input_schema: type[BaseModel]
    server_params: ServerParameters

    def __init__(self, input_schema: dict[str, Any], **kwargs):
        """
        Initializes the MCP tool with a given input schema and additional parameters.

        Args:
            input_schema (dict[str, Any]): JSON schema to define input fields.
            **kwargs
        """
        input_schema = MCPTool.create_input_schema(input_schema)
        super().__init__(input_schema=input_schema, **kwargs)

    @staticmethod
    def _map_json_type(json_type: str) -> type:
        """
        Maps MCP schema types to a corresponding Python type.

        Args:
            json_type (str): The JSON type as a string.

        Returns:
            type: The corresponding Python type.
        """
        return {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "object": dict,
            "array": list,
        }.get(json_type, str)

    @staticmethod
    def create_input_schema(schema_dict: dict[str, Any]):
        """
        Creates an input schema based on provided MCP schema.

        Args:
            schema_dict (dict[str, Any]): A JSON schema dictionary describing the tool's expected input.
        """
        fields = {}
        props = schema_dict.get("properties", {})
        required = set(schema_dict.get("required", []))

        for field_name, field_spec in props.items():
            field_type = MCPTool._map_json_type(field_spec.get("type", "string"))
            default = ... if field_name in required else None
            description = field_spec.get("description", None)
            fields[field_name] = (field_type, Field(default, description=description))

        return create_model(schema_dict.get("title", "MCPAdapterSchema"), **fields)

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
            async with await get_client(self.server_params) as (read, write):
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


class MCPAdapterTool(Node):
    """
    A tool that manages connections to MCP servers and initializes MCP tools.

    Attributes:
      group (Literal[NodeGroup.TOOLS]): Node group.
      name (str): Node name.
      description (str): Node description.
      mcp_tools (list): A list of initialized MCP tools.
      server_params (ServerParameters): The parameters for connecting to the MCP server.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "Mcp Adapter Tool"
    description: str = "The tool used to initialize available MCP tools based on provided server parameters."

    mcp_tools: list = []
    server_params: ServerParameters

    async def initialise_tools(self):
        """
        Initializes the MCP tool list from the client session.

        Returns:
            None
        """
        async with await get_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                self.mcp_tools = [
                    MCPTool(
                        name=tool.name,
                        description=tool.description,
                        input_schema=tool.inputSchema,
                        server_params=self.server_params,
                    )
                    for tool in tools.tools
                ]

        logger.info(f"Tool {self.name}: {len(self.mcp_tools)} MCP tools initialized from a server.")

    def get_mcp_tools(self) -> list[MCPTool]:
        """
        Synchronously fetches and initializes MCP tools if not already available.

        Returns:
            list[MCPTool]: A list of initialized MCPTool instances.
        """
        if is_called_from_async_context():
            with ThreadPoolExecutor() as executor:
                future = executor.submit(lambda: asyncio.run(self.get_mcp_tools_async()))
                return future.result()
        return asyncio.run(self.get_mcp_tools_async())

    async def get_mcp_tools_async(self) -> list[MCPTool]:
        """
        Asynchronously fetches and initializes MCP tools if not already available.

        Returns:
            list[MCPTool]: A list of initialized MCPTool instances.
        """
        if not self.mcp_tools:
            await self.initialise_tools()
        return self.mcp_tools

    def execute(self, **kwargs):
        """
        Disabled for the adapter tool. Use `get_mcp_tools()` to access individual tools.

        Raises:
            NotImplementedError: Always, because this method is not supported.
        """
        raise NotImplementedError("Use `get_mcp_tools()` to access individual tool instances.")
