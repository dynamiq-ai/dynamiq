import pytest

from dynamiq.nodes.agents import ReActAgent, ReflectionAgent, SimpleAgent
from dynamiq.nodes.llms import Anthropic, Gemini, OpenAI, WatsonX
from dynamiq.nodes.tools import E2BInterpreterTool, ScaleSerpTool, TavilyTool
from dynamiq.nodes.utils import Input, Output
from dynamiq.utils.node_generation import generate_data_from_schema, generate_node


@pytest.mark.parametrize(
    ("node", "params"),
    [
        (Input, {}),
        (Output, {}),
        (SimpleAgent, {"llms": {OpenAI: ["model1", "model2"]}, "tools": [E2BInterpreterTool]}),
        (ReActAgent, {"llms": {OpenAI: ["model1", "model2"]}, "tools": [E2BInterpreterTool]}),
        (ReflectionAgent, {"llms": {OpenAI: ["model1", "model2"]}, "tools": [E2BInterpreterTool]}),
        (OpenAI, {"models": ["model1", "model2"]}),
        (Gemini, {"models": ["model1", "model2"]}),
        (Anthropic, {"models": ["model1", "model2"]}),
        (WatsonX, {"models": ["model1", "model2"]}),
        (E2BInterpreterTool, {}),
        (ScaleSerpTool, {}),
        (TavilyTool, {}),
    ],
)
def test_vision_prompt_with_regular_url(node, params):
    """
    Tests if nodes can be generated from defined schemas
    """
    schema = node._generate_schema(**params)
    data = generate_data_from_schema(schema)

    try:
        generate_node(node, data, [])
    except Exception as e:
        pytest.fail(f"Failed to create User instance: {str(e)}")
