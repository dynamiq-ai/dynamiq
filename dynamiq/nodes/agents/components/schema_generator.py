"""Schema generation for Agent function calling and structured output modes."""

import types
from enum import Enum
from typing import Any, Callable, Union, get_args, get_origin

from dynamiq.nodes.agents.base import Agent as BaseAgent
from dynamiq.nodes.llms.gemini import Gemini
from dynamiq.nodes.node import Node

# Type mapping for schema generation
TYPE_MAPPING = {
    int: "integer",
    float: "float",
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
            },
            "required": ["thought", "answer"],
        },
    },
}


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
        for name, field in tool.input_schema.model_fields.items():
            if not field.json_schema_extra or field.json_schema_extra.get("is_accessible_to_agent", True):
                if get_origin(field.annotation) in (Union, types.UnionType):
                    type_str = str(field.annotation)
                else:
                    type_str = getattr(field.annotation, "__name__", str(field.annotation))

                if field.json_schema_extra and field.json_schema_extra.get("map_from_storage", False):
                    type_str = "tuple[str, ...]"

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

    if delegation_allowed and any(isinstance(tool, BaseAgent) for tool in tools):
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
                "required": ["thought", "action", "action_input"],
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
        types = []

        for param in params:
            if param is type(None):
                types.append("null")

            elif param_type := TYPE_MAPPING.get(param):
                types.append(param_type)

            elif issubclass(param, Enum):
                element_type = TYPE_MAPPING.get(filter_format_type(type(list(param.__members__.values())[0].value))[0])
                types.append(element_type)
                properties[name]["enum"] = [field.value for field in param.__members__.values()]

            elif getattr(param, "__origin__", None) is list:
                types.append("array")
                properties[name]["items"] = {"type": TYPE_MAPPING.get(param.__args__[0])}

            elif getattr(param, "__origin__", None) is dict:
                types.append("object")

        if len(types) == 1:
            properties[name]["type"] = types[0]
        elif len(types) > 1:
            properties[name]["type"] = types
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
        if isinstance(tool, BaseAgent):
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
                    "strict": True,
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
