import random
import types
from typing import Any, Union, get_args

from dynamiq.nodes import Node
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import BaseLLM
from dynamiq.nodes.node import ConnectionNode
from dynamiq.prompts import Message, MessageRole, Prompt
from dynamiq.serializers.loaders.yaml import WorkflowYAMLLoader
from dynamiq.utils import generate_uuid


def validate_input_transformer(messages: list[Message], node_data: Node) -> str:
    """
    Validates input transformer for Agents and LLM nodes

    Args:
        messages (list[Message]): Input message node accepts
        node_data (dict[str, Any]): Generated node data.

    Returns:
        str: Empty string if node information was successfully validated. Error message if not.
    """
    prompt = Prompt(messages=messages)
    required_parameters = prompt.get_required_parameters()

    provided_parameters = {element for element in list(node_data.input_transformer.selector.keys())}

    if required_parameters != provided_parameters:
        raise ValueError(
            f"Invalid parameters provided in node data. Required parameters: {list(required_parameters)}. "
            f"Provided parameters in InputTransformer {list(provided_parameters)}."
        )


def generate_integer(minimum: int, maximum: int) -> int:
    """
    Generate a random integer between the specified minimum and maximum values (inclusive).

    Args:
        minimum (int): The lower bound of the range.
        maximum (int): The upper bound of the range.

    Returns:
        int: A integer between minimum and maximum.
    """
    return random.randint(minimum, maximum)  # nosec


def generate_number(minimum: float, maximum: float) -> float:
    """
    Generate a random floating-point number between the specified minimum and maximum values.

    Args:
        minimum (float): The lower bound of the range.
        maximum (float): The upper bound of the range.

    Returns:
        float: A random float between minimum and maximum.
    """
    return random.uniform(minimum, maximum)  # nosec


def generate_boolean() -> bool:
    """
    Generate a random boolean value (True or False).

    Returns:
        bool: A randomly chosen boolean value.
    """
    return random.choice([False, True])  # nosec


def generate_data_from_schema(schema: dict[str, Any]) -> Any:
    """
    Recursive function that generates mock data based on provided schema.

    Args:
        schema (dict[str, Any]): Schema for data generation
    """
    schema_type = schema.get("type")

    if schema_type == "string":
        return schema.get("enum", ["mocked_data"])[0]

    elif schema_type == "integer":
        return generate_integer(minimum=schema.get("minimum", 0), maximum=schema.get("maximum", 100))
    elif schema_type == "number":
        return generate_number(minimum=schema.get("minimum", 0), maximum=schema.get("maximum", 1))

    elif schema_type == "boolean":
        return generate_boolean()

    elif schema_type == "object":
        obj = {}
        props = schema.get("properties", {})
        required = schema.get("required", [])
        for key, value_schema in props.items():
            if key in required:
                obj[key] = generate_data_from_schema(value_schema)
        return obj

    elif schema_type == "array":
        item_schema = schema.get("items", {})
        if list_elements := item_schema.get("anyOf"):
            return [generate_data_from_schema(list_elements[0])]
        return [generate_data_from_schema(item_schema)]

    elif any_object := schema.get("anyOf"):
        return generate_data_from_schema(any_object[0])

    return None


def add_connection(node: Node, data: dict[str, Any]) -> str:
    """Add connections generation data.

    Args:
        node (Node): Target node for adding a connection.
        data (dict[str, Any]): Generation data.

    Returns:
        str: New connection id.
    """
    node_connection_annotation = node.model_fields["connection"].annotation
    if type(node_connection_annotation) in (Union, types.UnionType):
        connection = next((x for x in get_args(node_connection_annotation) if x is not None), None)
    else:
        connection = node_connection_annotation

    connection_type = connection(api_key="").type
    connection_id = generate_uuid()

    if "connections" not in data:
        data["connections"] = {connection_id: {"type": connection_type, "api_key": ""}}
    else:
        data["connections"][connection_id] = {"type": connection_type, "api_key": ""}

    return connection_id


def generate_yaml_data(node_cls: type[Node], node_info: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """
    Generates WorkflowYamlLoader compatible data.

    Args:
        node_cls (type[Node]): Class of node to generate.
        node_info (dict[str, Any]): Node details.

    Returns:
        str: Generated id of the node.
        dict[str, Any]: WorkflowYamlLoader compatible data.
    """
    node_id = generate_uuid()

    if "input_transformer" in node_info:
        node_info["input_transformer"] = {
            "selector": {mapping["parameter"]: mapping["mapping"] for mapping in node_info["input_transformer"]}
        }

    data = {"nodes": {node_id: node_info}}
    if issubclass(node_cls, ConnectionNode):
        connection_id = add_connection(node_cls, data)
        data["nodes"][node_id]["connection"] = connection_id

    if issubclass(node_cls, Agent):
        for index, tool in enumerate(node_info["tools"]):
            entity_cls = WorkflowYAMLLoader.get_entity_by_type(tool["type"])
            connection_id = add_connection(entity_cls, data)
            data["nodes"][node_id]["tools"][index]["connection"] = connection_id

    return node_id, data


def generate_node(node_cls: type[Node], node_info: dict[str, Any], taken_names: list[str]):
    """
    Generates instance of node with unique name.

    Supported Nodes: Simple (non-nested) nodes and agents.

    Args:
        node_cls (type[Node]): Class of node to generate.
        node_info (dict[str, Any]): Node details.
        taken_names (list[str]): List of taken names.

    Returns:
        str: Generated id of the node.
        Node: Generated node.
    """

    node_id, data = generate_yaml_data(node_cls, node_info)
    result = WorkflowYAMLLoader.parse(data)
    node = result.nodes[node_id]

    if isinstance(node, Agent):
        node.llm.name = node.llm.name.lower().replace(" ", "-")
        for tool in node.tools:
            tool.name = tool.name.lower().replace(" ", "-")
    node.name = node.name.lower().replace(" ", "-")

    if node.input_transformer.selector:
        if isinstance(node, BaseLLM) and node.prompt:
            validate_input_transformer(node.prompt.messages, node)

        elif isinstance(node, Agent):
            validate_input_transformer([Message(role=MessageRole.USER, content="{{input}}")], node)
    if node.name in taken_names:
        raise ValueError(f"Name {node.name} is already taken.")

    return node_id, node
