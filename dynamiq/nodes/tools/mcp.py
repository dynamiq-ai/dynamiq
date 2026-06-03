import asyncio
import keyword
from dataclasses import field
from types import GenericAlias
from typing import Any, Literal, Union

from mcp import ClientSession
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, create_model

from dynamiq.connections import MCPSse, MCPStdio, MCPStreamableHTTP
from dynamiq.executors.context import ContextAwareThreadPoolExecutor
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils import is_called_from_async_context
from dynamiq.utils.logger import logger

NONE_TYPE = type(None)

JSON_SCHEMA_TYPE_MAPPING: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "object": dict[str, Any],
    "array": list[Any],
    "null": NONE_TYPE,
}

# Names that cannot be used directly as Pydantic field names: they shadow BaseModel
# attributes/methods (e.g. model_config, model_dump, schema, copy) and either break
# create_model or silently override model internals.
RESERVED_FIELD_NAMES = frozenset(dir(BaseModel))


def is_safe_field_name(name: str) -> bool:
    """Whether a JSON Schema property name can be used as-is as a Pydantic field name."""
    return name.isidentifier() and not keyword.iskeyword(name) and name not in RESERVED_FIELD_NAMES


def make_safe_field_name(name: str, used: set[str]) -> str:
    """Derive a unique, valid Python field name for an unsafe JSON Schema property name."""
    candidate = "".join(char if char.isalnum() else "_" for char in name).strip("_")
    if not candidate or candidate[0].isdigit() or not is_safe_field_name(candidate):
        candidate = f"field_{candidate}".rstrip("_")
    base = candidate
    index = 1
    while candidate in used:
        candidate = f"{base}_{index}"
        index += 1
    return candidate


def rename_keys_recursive(data: dict[str, Any] | list[Any], key_map: dict[str, str]) -> Any:
    """
    Recursively renames keys in a nested dictionary based on a provided key mapping.
    """
    if isinstance(data, dict):
        return {key_map.get(key, key): rename_keys_recursive(value, key_map) for key, value in data.items()}
    elif isinstance(data, list):
        return [rename_keys_recursive(item, key_map) for item in data]
    return data


def extract_text_from_mcp_content(content: list[Any]) -> str:
    """Join the text of the TextContent blocks in an MCP result's content list."""
    return "\n".join(block.text for block in content if getattr(block, "type", None) == "text")


def get_json_schema_definitions(schema: dict[str, Any]) -> dict[str, Any]:
    definitions: dict[str, Any] = {}
    for key in ("$defs", "definitions"):
        value = schema.get(key)
        if isinstance(value, dict):
            definitions.update(value)
    return definitions


def make_model_name(name: str | None, fallback: str = "MCPToolSchema") -> str:
    if not name:
        return fallback
    parts = [part for part in "".join(char if char.isalnum() else " " for char in name).title().split() if part]
    model_name = "".join(parts)
    if not model_name or model_name[0].isdigit():
        return fallback
    return model_name


def get_union_type(types: list[Any]) -> Any:
    unique_types = []
    for type_ in types:
        if type_ not in unique_types:
            unique_types.append(type_)
    if not unique_types:
        return Any
    if len(unique_types) == 1:
        return unique_types[0]
    return Union.__getitem__(tuple(unique_types))


def get_literal_type(values: list[Any]) -> Any:
    return Literal.__getitem__(tuple(values))


def get_list_annotation(item_type: Any) -> Any:
    return GenericAlias(list, (item_type,))


def get_dict_annotation(value_type: Any) -> Any:
    return GenericAlias(dict, (str, value_type))


def resolve_json_schema_ref(ref: str, definitions: dict[str, Any]) -> dict[str, Any]:
    prefix = "#/$defs/"
    definitions_prefix = "#/definitions/"
    if ref.startswith(prefix):
        value = definitions.get(ref.removeprefix(prefix))
        return value if isinstance(value, dict) else {}
    if ref.startswith(definitions_prefix):
        value = definitions.get(ref.removeprefix(definitions_prefix))
        return value if isinstance(value, dict) else {}
    return {}


def merge_all_of(members: list[Any], definitions: dict[str, Any], seen_refs: frozenset[str]) -> dict[str, Any]:
    """Shallow-merge the subschemas of an ``allOf`` into a single schema dict (intersection)."""
    merged: dict[str, Any] = {}
    merged_properties: dict[str, Any] = {}
    merged_required: list[Any] = []
    for member in members:
        if not isinstance(member, dict):
            continue
        ref = member.get("$ref")
        if isinstance(ref, str):
            if ref in seen_refs:
                continue  # cyclic member already being resolved upstream
            member = resolve_json_schema_ref(ref, definitions)
        properties = member.get("properties")
        if isinstance(properties, dict):
            merged_properties.update(properties)
        required = member.get("required")
        if isinstance(required, list):
            merged_required.extend(required)
        for key, value in member.items():
            if key not in ("properties", "required", "$ref", "allOf"):
                merged.setdefault(key, value)
    if merged_properties:
        merged["properties"] = merged_properties
        merged["type"] = "object"
    if merged_required:
        merged["required"] = merged_required
    return merged


def json_schema_to_type(
    prop: dict[str, Any], name: str, definitions: dict[str, Any], seen_refs: frozenset[str] = frozenset()
) -> Any:
    ref = prop.get("$ref")
    if isinstance(ref, str):
        if ref in seen_refs:
            # A self-recursive definition (e.g. a tree node whose children are the same node).
            # We type the cycle-closing point permissively instead of building a true recursive
            # model: the latter needs forward references + model_rebuild(), the same lazy-annotation
            # path that caused the original "name 'Optional' is not defined" crash. The data still
            # round-trips and the MCP server validates it on receipt. Non-recursive reuse of a
            # definition (sibling or nested) is unaffected and still builds a fully-typed model.
            return dict[str, Any]
        resolved = resolve_json_schema_ref(ref, definitions)
        if not resolved:
            logger.warning(f"MCP tool schema: could not resolve $ref '{ref}'; falling back to a permissive type.")
            return Any
        prop = resolved
        seen_refs = seen_refs | {ref}

    enum_values = prop.get("enum")
    if isinstance(enum_values, list) and enum_values:
        return get_literal_type(enum_values)

    schema_all_of = prop.get("allOf")
    if isinstance(schema_all_of, list) and schema_all_of:
        member_refs = {m["$ref"] for m in schema_all_of if isinstance(m, dict) and isinstance(m.get("$ref"), str)}
        merged = merge_all_of(schema_all_of, definitions, seen_refs)
        return json_schema_to_type(merged, name, definitions, seen_refs | member_refs)

    schema_options = prop.get("anyOf") or prop.get("oneOf")
    if isinstance(schema_options, list):
        return get_union_type(
            [
                json_schema_to_type(option, name, definitions, seen_refs)
                for option in schema_options
                if isinstance(option, dict)
            ]
        )

    schema_type = prop.get("type")
    if isinstance(schema_type, list):
        return get_union_type(
            [
                json_schema_to_type({**prop, "type": type_}, name, definitions, seen_refs)
                for type_ in schema_type
                if isinstance(type_, str)
            ]
        )

    if schema_type == "object" or prop.get("properties"):
        if prop.get("properties"):
            title = prop.get("title")
            return create_input_schema_from_json_schema(
                prop,
                make_model_name(title if isinstance(title, str) else name),
                definitions,
                seen_refs,
            )
        additional_props = prop.get("additionalProperties")
        if isinstance(additional_props, dict):
            return get_dict_annotation(json_schema_to_type(additional_props, f"{name}Value", definitions, seen_refs))
        return dict[str, Any]

    if schema_type == "array":
        items = prop.get("items", {})
        item_schema = items if isinstance(items, dict) else {}
        return get_list_annotation(json_schema_to_type(item_schema, f"{name}Item", definitions, seen_refs))

    return JSON_SCHEMA_TYPE_MAPPING.get(schema_type, Any)


def create_input_schema_from_json_schema(
    schema_dict: dict[str, Any],
    model_name: str = "MCPToolSchema",
    definitions: dict[str, Any] | None = None,
    seen_refs: frozenset[str] = frozenset(),
) -> type[BaseModel]:
    definitions = {**(definitions or {}), **get_json_schema_definitions(schema_dict)}
    required = schema_dict.get("required", [])
    required_fields = set(required if isinstance(required, list) else [])
    fields: dict[str, tuple[Any, Any]] = {}

    properties = schema_dict.get("properties", {})
    if not isinstance(properties, dict):
        properties = {}

    used_field_names: set[str] = set()
    for name, prop in properties.items():
        if not isinstance(prop, dict):
            continue
        field_type = json_schema_to_type(prop, name, definitions, seen_refs)
        default = ... if name in required_fields else prop.get("default", None)
        if name not in required_fields:
            field_type = get_union_type([field_type, NONE_TYPE])

        raw_description = prop.get("description")
        description = raw_description if isinstance(raw_description, str) else None
        enum_values = prop.get("enum")
        if isinstance(enum_values, list) and enum_values:
            enum_description = f" Allowed values: {', '.join(map(str, enum_values))}."
            description = (description or "").rstrip() + enum_description

        if is_safe_field_name(name) and name not in used_field_names:
            field_name = name
            alias = None
        else:
            field_name = make_safe_field_name(name, used_field_names)
            alias = name
        used_field_names.add(field_name)

        fields[field_name] = (field_type, Field(default, description=description, alias=alias))

    return create_model(
        model_name,
        __config__=ConfigDict(populate_by_name=True, protected_namespaces=()),
        **fields,
    )


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
        return create_input_schema_from_json_schema(
            schema_dict,
            "MCPToolSchema",
        )

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
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump(by_alias=True)}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        input_dict = input_data.model_dump(by_alias=True)

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

        if result.isError:
            error_text = extract_text_from_mcp_content(result.content)
            raise ToolExecutionException(
                f"Tool '{self.name}' returned an error from the MCP server: "
                f"{error_text or 'unknown error'}. Please analyze the error and take appropriate action.",
                recoverable=True,
            )

        raw_response = result.model_dump(exclude_none=True)

        if not self.is_optimized_for_agents:
            content = raw_response
        elif result.structuredContent is not None:
            content = result.structuredContent
        else:
            content = extract_text_from_mcp_content(result.content)

        logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(content)[:200]}...")
        output = {"content": content}
        if self.is_optimized_for_agents:
            output["raw_response"] = raw_response
        return output


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
    name: str = "mcp"
    description: str = """Model Context Server integration for
    dynamic tool discovery and external service connectivity.

Key Capabilities:
- Dynamic tool discovery and initialization from MCP servers
- Support for multiple connection types (SSE, stdio, HTTP streaming)
- Automatic schema generation from MCP tool definitions
- Tool filtering and selection with include/exclude lists

Usage Strategy:
- Integrate with external services via MCP protocol
- Access remote tools and APIs through standardized interface
- Build distributed tool ecosystems across services
- Extend workflow capabilities with external service tools
"""  # noqa: E501
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
        try:
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
                            is_optimized_for_agents=self.is_optimized_for_agents,
                        )

            logger.info(f"Tool {self.name}: {len(self._mcp_tools)} MCP tools initialized from a server.")
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to initialize session. Error: {str(e)}")
            raise ToolExecutionException(f"Tool {self.name} - {self.id}: failed to initialize session. Error: {str(e)}")

    def get_mcp_tools(self, select_all: bool = False) -> list[MCPTool]:
        """
        Synchronously fetches and initializes MCP tools if not already available.

        Args:
            select_all (bool): If True, returns all tools regardless of filtering.

        Returns:
            list[MCPTool]: A list of initialized MCPTool instances.
        """
        if is_called_from_async_context():
            with ContextAwareThreadPoolExecutor() as executor:
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
