import pytest

from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import Anthropic, Gemini, OpenAI, WatsonX
from dynamiq.nodes.node import ConnectionNode
from dynamiq.nodes.tools import E2BInterpreterTool, ScaleSerpTool, TavilyTool
from dynamiq.nodes.utils import Input, Output
from dynamiq.serializers.loaders.yaml import WorkflowYAMLLoader
from dynamiq.utils.node_generation import add_connection, generate_data_from_schema, generate_node


@pytest.mark.parametrize(
    ("node", "params"),
    [
        (Input, {}),
        (Output, {}),
        (Agent, {"llms": {OpenAI: ["model1", "model2"]}, "tools": [E2BInterpreterTool]}),
        (Agent, {"llms": {OpenAI: ["model1", "model2"]}, "tools": [E2BInterpreterTool]}),
        (Agent, {"llms": {OpenAI: ["model1", "model2"]}, "tools": [E2BInterpreterTool]}),
        (OpenAI, {"models": ["model1", "model2"]}),
        (Gemini, {"models": ["model1", "model2"]}),
        (Anthropic, {"models": ["model1", "model2"]}),
        (WatsonX, {"models": ["model1", "model2"]}),
        (E2BInterpreterTool, {}),
        (ScaleSerpTool, {}),
        (TavilyTool, {}),
    ],
)
def test_nodes_schema_generation(node, params):
    """
    Tests if nodes can be generated from defined schemas.
    """
    schema = node._generate_json_schema(**params)
    data = generate_data_from_schema(schema)

    try:
        _, node_instance = generate_node(node, data, [])
        assert isinstance(node_instance, node)

        instance_data = node_instance.to_dict(by_alias=True)

        node_id = "test_node_id"
        data = {"nodes": {node_id: {**instance_data}}}
        if issubclass(node, ConnectionNode):
            connection_id = add_connection(node, data)
            data["nodes"][node_id]["connection"] = connection_id

        if issubclass(node, Agent):
            for index, tool in enumerate(instance_data["tools"]):
                entity_cls = WorkflowYAMLLoader.get_entity_by_type(tool["type"])
                connection_id = add_connection(entity_cls, data)
                data["nodes"][node_id]["tools"][index]["connection"] = connection_id

            entity_cls = WorkflowYAMLLoader.get_entity_by_type(instance_data["llm"]["type"])
            connection_id = add_connection(entity_cls, data)
            data["nodes"][node_id]["llm"]["connection"] = connection_id

        parsed_result = WorkflowYAMLLoader.parse(data)
        reproduced_node = parsed_result.nodes[node_id]

        assert isinstance(reproduced_node, node)

    except ValueError as e:
        pytest.fail(f"Failed to create Node {node.__name__} instance: {str(e)}")
