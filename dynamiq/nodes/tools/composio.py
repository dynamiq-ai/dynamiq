import json
from typing import Any, Literal, Optional
from urllib.parse import urljoin

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, create_model
from requests import RequestException

from dynamiq.connections import Composio as ComposioConnection
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

JSON_TYPE_MAPPING = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}


class _BaseComposioInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _resolve_field_type(schema: dict[str, Any]) -> type[Any]:
    schema_type = schema.get("type")

    if isinstance(schema_type, list):
        schema_type = next((item for item in schema_type if item != "null"), None)

    if schema.get("enum"):
        values = tuple(schema["enum"])
        if values:
            from typing import Literal as _Literal

            return _Literal[values]  # type: ignore[index]

    if schema_type == "array":
        items = schema.get("items")
        item_type = _resolve_field_type(items) if isinstance(items, dict) else Any
        return list[item_type]
    if schema_type == "object":
        return dict[str, Any]
    if schema_type in JSON_TYPE_MAPPING:
        return JSON_TYPE_MAPPING[schema_type]
    return Any


def _field_kwargs(schema: dict[str, Any]) -> dict[str, Any]:
    json_extra = {key: value for key in ("enum", "examples", "format") if (value := schema.get(key))}

    kwargs: dict[str, Any] = {}
    if description := schema.get("description"):
        kwargs["description"] = description
    if json_extra:
        kwargs["json_schema_extra"] = json_extra
    return kwargs


def _build_input_model(tool_schema: dict[str, Any]) -> type[BaseModel]:
    parameters = tool_schema.get("input_parameters") or {}
    properties = parameters.get("properties") or {}
    required_fields = set(parameters.get("required") or [])

    fields: dict[str, tuple[type[Any], Field]] = {}
    for name, schema in properties.items():
        field_type = _resolve_field_type(schema)
        kwargs = _field_kwargs(schema)
        default = schema.get("default")
        is_nullable = schema.get("nullable") is True or schema.get("type") == "null"
        is_required = name in required_fields and default is None and not is_nullable

        if not is_required:
            annotated_type = Optional[field_type]
            fields[name] = (annotated_type, Field(default=default, **kwargs))
        else:
            fields[name] = (field_type, Field(..., **kwargs))

    slug = tool_schema.get("slug", "composio_tool")
    if not isinstance(slug, str):
        slug = getattr(slug, "value", str(slug))
    model_name = f"ComposioInput_{slug.replace('.', '_').replace('-', '_')}"
    return create_model(model_name, __base__=_BaseComposioInput, **fields)


class Composio(ConnectionNode):
    """Tool node for executing Composio tools through the public API."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "Composio Tool"
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    connection: ComposioConnection | None = None

    tool_slug: str
    version: str | None = None
    description: str = ""
    connected_account_id: str | None = None
    entity_id: str | None = None
    input_schema: type[BaseModel] | None = None
    url: str = Field(
        default="https://backend.composio.dev/api/v3",
    )
    timeout: float = 30.0

    _tool_schema: dict[str, Any] | None = PrivateAttr(default=None)
    _base_description: str | None = PrivateAttr(default=None)

    def __init__(self, **kwargs):
        base_description = kwargs.get("description") or None
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = ComposioConnection()
        super().__init__(**kwargs)
        self._base_description = base_description

    @property
    def to_dict_exclude_params(self):
        parent = super().to_dict_exclude_params.copy()
        parent.update({"input_schema": True})
        return parent

    def init_components(self, connection_manager: ConnectionManager | None = None):
        super().init_components(connection_manager)
        self._ensure_schema()

    def _connection_url(self) -> str:
        if "url" in self.model_fields_set or self.connection is None:
            return self.url
        connection_url = getattr(self.connection, "url", None)
        return connection_url or self.url

    def _connection_timeout(self) -> float:
        if "timeout" in self.model_fields_set or self.connection is None:
            return self.timeout
        connection_timeout = getattr(self.connection, "timeout", None)
        if connection_timeout is None:
            return self.timeout
        return connection_timeout

    def _slug(self) -> str:
        value = self.tool_slug
        if isinstance(value, str):
            return value
        return getattr(value, "value", str(value))

    def _ensure_schema(self) -> None:
        if self.input_schema is not None and self._tool_schema is not None:
            return

        tool_schema = self._get_tool_schema()
        self._tool_schema = tool_schema
        self.input_schema = _build_input_model(tool_schema)
        self.description = self._generate_description(tool_schema)
        if self.name == "Composio Tool":
            self.name = f"Composio {self._slug()}"

    def _get_tool_schema(self) -> dict[str, Any]:
        if self._tool_schema is not None:
            return self._tool_schema

        slug = self._slug()
        params = {"version": self.version} if self.version else None
        data = self._request_json("get", f"/tools/{slug}", params=params)
        return data

    def _generate_description(self, tool_schema: dict[str, Any]) -> str:
        schema_fields = self.input_schema.model_fields if self.input_schema else {}

        slug = tool_schema.get("slug") or self._slug()
        docs_url = tool_schema.get("docs_url") or tool_schema.get("documentation_url")
        base_line = self._base_description or tool_schema.get("description") or f"Execute Composio tool '{slug}'."

        lines: list[str] = [base_line]
        if docs_url:
            lines.append(f"Documentation: {docs_url}")
        lines.append(f"Tool Slug: {slug}")
        lines.append("")

        required = [name for name, field in schema_fields.items() if field.is_required() is not False]
        lines.append("Required Parameters:")
        if required:
            for name in sorted(required):
                field = schema_fields[name]
                description = field.description or "No description provided."
                lines.append(f"- {name}: {description}")
        else:
            lines.append("- None")

        lines.append("")

        optional = [name for name, field in schema_fields.items() if field.is_required() is False]
        lines.append("Optional Parameters:")
        if optional:
            for name in sorted(optional):
                field = schema_fields[name]
                description = field.description or "No description provided."
                lines.append(f"- {name}: {description}")
        else:
            lines.append("- None")

        return "\n".join(lines)

    def execute(
        self, input_data: BaseModel | dict[str, Any], config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        self._ensure_schema()

        if isinstance(input_data, dict):
            if self.input_schema is None:
                raise ToolExecutionException("Composio tool schema is not initialized", recoverable=True)
            input_data = self.input_schema.model_validate(input_data)

        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        payload: dict[str, Any] = {"arguments": {k: v for k, v in input_data.model_dump().items() if v is not None}}

        entity_id = self.entity_id
        if entity_id is None and self.connection is not None:
            entity_id = self.connection.entity_id

        connected_account = self.connected_account_id
        if connected_account is None and self.connection is not None:
            connected_account = self.connection.connected_account_id

        if entity_id is not None:
            payload["entity_id"] = entity_id
        if connected_account is not None:
            payload["connected_account_id"] = connected_account
        if self.version:
            payload["version"] = self.version

        slug = self._slug()
        result = self._request_json("post", f"/tools/execute/{slug}", json=payload)

        if not result.get("successful", True):
            error_message = result.get("error") or "Composio tool execution failed"
            raise ToolExecutionException(
                f"Failed to execute Composio tool '{self.tool_slug}': {error_message}", recoverable=True
            )

        data = result.get("data")
        if self.is_optimized_for_agents and isinstance(data, (dict, list)):
            content = json.dumps(data)
        else:
            content = data

        return {"content": content}

    def _request_json(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        if self.client is None:
            raise ToolExecutionException("Composio client is not initialized", recoverable=True)

        base_url = self._connection_url()
        request_url = urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))

        request_kwargs = {**kwargs}
        request_kwargs.setdefault("timeout", self._connection_timeout())

        try:
            response = self.client.request(method, request_url, **request_kwargs)
        except Exception as exc:  # pragma: no cover - network errors
            logger.error(f"Tool {self.name} - Error calling Composio API: {exc}")
            raise ToolExecutionException("Composio API request failed", recoverable=True) from exc

        try:
            response.raise_for_status()
        except RequestException as exc:
            logger.error(f"Tool {self.name} - API responded with error: {exc}")
            raise ToolExecutionException("Composio API responded with an error", recoverable=True) from exc

        try:
            return response.json()
        except ValueError as exc:
            raise ToolExecutionException("Invalid JSON returned from Composio API", recoverable=True) from exc
