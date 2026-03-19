"""Schema generation for Agent function calling and structured output modes."""

import types
from enum import Enum
from typing import Any, Callable, Literal, Union, get_args, get_origin

from pydantic import BaseModel

from dynamiq.nodes.llms.gemini import Gemini
from dynamiq.nodes.node import Node
from dynamiq.nodes.tools.agent_tool import SubAgentTool

# Type mapping for schema generation
TYPE_MAPPING = {
    int: "integer",
    float: "number",
    bool: "boolean",
    str: "string",
    dict: "object",
}

# Final answer function schema for function calling mode
FINAL_ANSWER_FUNCTION_SCHEMA = {
    "type": "function",
    "strict": True,
    "function": {
        "name": "provide_final_answer",
        "description": "Function should be called when if you can answer the initial request"
        " or if there is not request at all.",
        "parameters": {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "Your reasoning about why you can answer original question.",
                },
                "answer": {"type": "string", "description": "Answer on initial request."},
                "output_files": {
                    "type": "string",
                    "description": "Optional comma-separated file paths to return. Empty string if none.",
                },
            },
            "required": ["thought", "answer", "output_files"],
        },
    },
}


PRIORITY_FIELDS = ("brief",)


def _reorder_fields(fields: dict) -> list[tuple[str, Any]]:
    """Reorder fields so that priority fields (e.g. brief) come first."""
    priority = [(k, v) for k, v in fields.items() if k in PRIORITY_FIELDS]
    rest = [(k, v) for k, v in fields.items() if k not in PRIORITY_FIELDS]
    return priority + rest


def generate_input_formats(tools: list[Node], sanitize_tool_name: Callable[[str], str]) -> str:
    """
    Generate formatted input descriptions for each tool.

    Args:
        tools: List of tools to generate input formats for
        sanitize_tool_name: Function to sanitize tool names

    Returns:
        Formatted string describing input parameters for each tool
    """
    input_formats = []
    for tool in tools:
        params = []
        for name, field in _reorder_fields(tool.input_schema.model_fields):
            if not field.json_schema_extra or field.json_schema_extra.get("is_accessible_to_agent", True):
                args = get_args(field.annotation)
                if get_origin(field.annotation) in (Union, types.UnionType):
                    type_str = str(field.annotation)
                elif field.json_schema_extra and field.json_schema_extra.get("map_from_storage", False):
                    type_str = "tuple[str, ...]"
                elif args and hasattr(args[0], "model_fields") and get_origin(field.annotation) is list:
                    nested_fields = [
                        f"{fn}: {getattr(fi.annotation, '__name__', str(fi.annotation))} - {fi.description or ''}"
                        for fn, fi in args[0].model_fields.items()
                    ]
                    type_str = f"list[{{{', '.join(nested_fields)}}}, ...]"
                else:
                    type_str = getattr(field.annotation, "__name__", str(field.annotation))

                description = field.description or "No description"
                params.append(f"{name} ({type_str}): {description}")
        if params:
            input_formats.append(f" - {sanitize_tool_name(tool.name)}\n \t* " + "\n\t* ".join(params))
    return "\n".join(input_formats)


def generate_structured_output_schemas(
    tools: list[Node], sanitize_tool_name: Callable[[str], str], delegation_allowed: bool
) -> dict:
    """
    Generate schema for structured output mode.

    Args:
        tools: List of tools to generate schema for
        sanitize_tool_name: Function to sanitize tool names
        delegation_allowed: Whether delegation is allowed

    Returns:
        Dictionary containing the structured output schema
    """
    tool_names = [sanitize_tool_name(tool.name) for tool in tools]

    action_input_description = "Input for chosen action."

    if delegation_allowed and any(isinstance(tool, SubAgentTool) for tool in tools):
        action_input_description += (
            ' For agent tools, include {"input": "<subtask>", "delegate_final": true} '
            "to return that agent's response directly as the final answer."
        )

    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "plan_next_action",
            "strict": True,
            "schema": {
                "type": "object",
                "required": ["thought", "action", "action_input", "output_files"],
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "Your reasoning about the next step.",
                    },
                    "action": {
                        "type": "string",
                        "description": f"Next action to make (choose from [{tool_names}, finish]).",
                    },
                    "action_input": {
                        "type": "string",
                        "description": action_input_description,
                    },
                    "output_files": {
                        "type": "string",
                        "description": (
                            "Comma-separated file paths to return when action is finish. Empty string otherwise."
                        ),
                    },
                },
                "additionalProperties": False,
            },
        },
    }

    return schema


def filter_format_type(param_annotation: Any) -> list[str]:
    """
    Filters proper type for a function calling schema.

    Args:
        param_annotation: Parameter annotation

    Returns:
        List of parameter types that describe provided annotation
    """
    if get_origin(param_annotation) in (Union, types.UnionType):
        return get_args(param_annotation)

    return [param_annotation]


def _resolve_type_schema(param: Any) -> dict[str, Any] | None:
    """Return a JSON Schema fragment for a single type.

    ``BaseModel`` subclasses are expanded into proper object schemas with
    ``properties`` so the LLM produces correctly structured output.
    Generic ``dict`` types become bare ``{"type": "object"}``.

    Tools whose schemas contain bare objects automatically get
    ``strict: false`` via ``_is_strict_compatible``.
    """
    if param is type(None):
        return {"type": "null"}

    if param_type := TYPE_MAPPING.get(param):
        return {"type": param_type}

    if isinstance(param, type) and issubclass(param, Enum):
        element_type = TYPE_MAPPING.get(
            filter_format_type(type(list(param.__members__.values())[0].value))[0],
            "string",
        )
        return {"type": element_type, "enum": [m.value for m in param.__members__.values()]}

    origin = get_origin(param)

    if origin in (Union, types.UnionType):
        args = [a for a in get_args(param) if a is not type(None)]
        for arg in args:
            resolved = _resolve_type_schema(arg)
            if resolved is not None:
                return resolved
        return {"type": "string"}

    if origin is Literal:
        values = list(get_args(param))
        lit_type = type(values[0]) if values else str
        return {"type": TYPE_MAPPING.get(lit_type, "string"), "enum": values}

    if origin is list:
        inner_args = get_args(param)
        if not inner_args:
            return {"type": "array", "items": {"type": "string"}}
        inner_schema = _resolve_type_schema(inner_args[0])
        return {"type": "array", "items": inner_schema or {"type": "string"}}

    if origin is dict:
        return {"type": "object"}

    if isinstance(param, type) and issubclass(param, BaseModel):
        return _basemodel_to_schema(param)

    return None


def _basemodel_to_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Build an object schema from a Pydantic model with explicit properties.

    All fields are listed in ``required`` and ``additionalProperties`` is set
    to ``False`` so the schema satisfies OpenAI strict-mode constraints.
    """
    properties: dict[str, Any] = {}
    for name, field in model.model_fields.items():
        schema = _resolve_type_schema(field.annotation)
        if schema is None:
            schema = {"type": "string"}
        if field.description:
            schema["description"] = field.description
        properties[name] = schema
    result: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": list(properties.keys()),
        "additionalProperties": False,
    }
    return result


def _is_strict_compatible(schema: Any) -> bool:
    """Return ``False`` if the schema contains a bare ``{"type": "object"}``
    without ``properties`` — which OpenAI strict mode rejects."""
    if not isinstance(schema, dict):
        return True
    if schema.get("type") == "object" and "properties" not in schema:
        return False
    for value in schema.values():
        if isinstance(value, dict) and not _is_strict_compatible(value):
            return False
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and not _is_strict_compatible(item):
                    return False
    return True


def generate_property_schema(properties: dict, name: str, field: Any) -> None:
    """
    Generate property schema for a field in function calling mode.

    Args:
        properties: Dictionary to store the generated property schema
        name: Name of the property
        field: Field object containing metadata
    """
    if not field.json_schema_extra or field.json_schema_extra.get("is_accessible_to_agent", True):
        description = field.description or "No description."

        description += f" Defaults to: {field.default}." if field.default and not field.is_required() else ""
        params = filter_format_type(field.annotation)

        properties[name] = {"description": description}
        schemas = [s for p in params if (s := _resolve_type_schema(p)) is not None]

        if len(schemas) == 1:
            properties[name].update(schemas[0])
        elif len(schemas) > 1:
            non_null = [s for s in schemas if s != {"type": "null"}]
            if len(non_null) == 1:
                properties[name].update(non_null[0])
            else:
                properties[name].update(non_null[0] if non_null else schemas[0])
        else:
            properties[name]["type"] = "string"


def generate_function_calling_schemas(
    tools: list[Node], delegation_allowed: bool, sanitize_tool_name: Callable[[str], str], llm: Any
) -> list[dict]:
    """
    Generate schemas for function calling mode.

    Args:
        tools: List of tools to generate schemas for
        delegation_allowed: Whether delegation is allowed
        sanitize_tool_name: Function to sanitize tool names
        llm: The LLM instance

    Returns:
        List of function calling schemas for all tools
    """
    schemas = [FINAL_ANSWER_FUNCTION_SCHEMA]

    for tool in tools:
        # Agent tools: accept action_input as a JSON string to avoid nested schema issues.
        if isinstance(tool, SubAgentTool):
            agent_action_input_description = "JSON string containing the agent's inputs "
            if delegation_allowed:
                agent_action_input_description += '(e.g., {"input": "<subtask>", "delegate_final": true}).'
            else:
                agent_action_input_description += '(e.g., {"input": "<subtask>"}).'

            schema = {
                "type": "function",
                "function": {
                    "name": sanitize_tool_name(tool.name),
                    "description": tool.description[:1024],
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "thought": {
                                "type": "string",
                                "description": "Your reasoning about using this tool.",
                            },
                            "action_input": {
                                "type": "string",
                                "description": agent_action_input_description,
                            },
                        },
                        "additionalProperties": False,
                        "required": ["thought", "action_input"],
                    },
                    "strict": True,
                },
            }

            schemas.append(schema)
            continue

        properties = {}
        input_params = tool.input_schema.model_fields.items()
        if list(input_params) and not isinstance(llm, Gemini):
            for name, field in tool.input_schema.model_fields.items():
                generate_property_schema(properties, name, field)

            use_strict = _is_strict_compatible(properties)

            schema = {
                "type": "function",
                "function": {
                    "name": sanitize_tool_name(tool.name),
                    "description": tool.description[:1024],
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "thought": {
                                "type": "string",
                                "description": "Your reasoning about using this tool.",
                            },
                            "action_input": {
                                "type": "object",
                                "description": "Input for the selected tool",
                                "properties": properties,
                                "required": list(properties.keys()),
                                "additionalProperties": False,
                            },
                        },
                        "additionalProperties": False,
                        "required": ["thought", "action_input"],
                    },
                    "strict": use_strict,
                },
            }

            schemas.append(schema)

        else:
            schema = {
                "type": "function",
                "function": {
                    "name": sanitize_tool_name(tool.name),
                    "description": tool.description[:1024],
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "thought": {
                                "type": "string",
                                "description": "Your reasoning about using this tool.",
                            },
                            "action_input": {
                                "type": "string",
                                "description": "Input for the selected tool in JSON string format.",
                            },
                        },
                        "additionalProperties": False,
                        "required": ["thought", "action_input"],
                    },
                    "strict": True,
                },
            }

            schemas.append(schema)

    return schemas
