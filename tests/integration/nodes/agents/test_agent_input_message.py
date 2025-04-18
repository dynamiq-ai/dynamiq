import uuid

import pytest

from dynamiq import connections
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.agents.reflection import ReflectionAgent
from dynamiq.nodes.agents.simple import SimpleAgent
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.types import InferenceMode
from dynamiq.prompts import (
    Message,
    MessageRole,
    VisionMessage,
    VisionMessageImageContent,
    VisionMessageImageURL,
    VisionMessageTextContent,
)

REQUEST_TEMPLATE = "Request: {{request}}"
URL_TEMPLATE = "{{url}}"
CONTEXT_TEMPLATE = "{{context}}"

REQUEST = "request_placeholder"
URL = "url_placeholder"
CONTEXT = "context_placeholder"


@pytest.fixture
def model():
    connection = connections.OpenAI(
        id=str(uuid.uuid4()),
        api_key="api-key",
    )
    return OpenAI(
        name="OpenAI",
        model="gpt-4o-mini",
        connection=connection,
    )


def create_react_agent(llm, inference_mode, input_message=None, role=None):
    """ReActAgent"""
    return ReActAgent(
        name="Test Agent", llm=llm, tools=[], input_message=input_message, role=role, inference_mode=inference_mode
    )


def create_simple_agent(llm, inference_mode, input_message=None, role=None):
    """SimpleAgent"""
    return SimpleAgent(
        name="Test Agent", llm=llm, inference_mode=inference_mode, role=role, input_message=input_message
    )


def create_reflection_agent(llm, inference_mode, input_message=None, role=None):
    """ReflectionAgent"""
    return ReflectionAgent(
        name="Test Agent", llm=llm, inference_mode=inference_mode, role=role, input_message=input_message
    )


@pytest.mark.parametrize(
    ("inference_mode"),
    [
        (InferenceMode.DEFAULT),
        (InferenceMode.XML),
        (InferenceMode.STRUCTURED_OUTPUT),
        (InferenceMode.FUNCTION_CALLING),
    ],
)
def test_simple_agent_workflow(model, inference_mode):
    react_agent = create_react_agent(model, inference_mode)

    react_agent.run(
        input_data={"input": REQUEST},
    )
    expected_result = Message(content=REQUEST, role=MessageRole.USER)
    assert react_agent._prompt.messages[1] == expected_result

    simple_agent = create_simple_agent(model, inference_mode)
    simple_agent.run(
        input_data={"input": REQUEST},
    )
    assert simple_agent._prompt.messages[1] == expected_result

    reflection_agent = create_reflection_agent(model, inference_mode)
    reflection_agent.run(
        input_data={"input": REQUEST},
    )
    assert reflection_agent._prompt.messages[1] == expected_result


@pytest.mark.parametrize(
    ("inference_mode"),
    [
        (InferenceMode.DEFAULT),
        (InferenceMode.XML),
        (InferenceMode.STRUCTURED_OUTPUT),
        (InferenceMode.FUNCTION_CALLING),
    ],
)
def test_custom_agent_workflow(model, inference_mode):
    input_message = Message(content=REQUEST_TEMPLATE, role=MessageRole.USER)

    react_agent = create_react_agent(model, inference_mode, input_message, role=CONTEXT_TEMPLATE)

    react_agent.run(
        input_data={"request": REQUEST, "context": CONTEXT},
    )
    expected_result = input_message.format_message(request=REQUEST)
    assert react_agent._prompt.messages[1] == expected_result
    assert CONTEXT in react_agent._prompt.messages[0].content

    simple_agent = create_simple_agent(model, inference_mode, input_message, role=CONTEXT_TEMPLATE)
    simple_agent.run(
        input_data={"request": REQUEST, "context": CONTEXT},
    )
    assert simple_agent._prompt.messages[1] == expected_result
    assert CONTEXT in simple_agent._prompt.messages[0].content

    reflection_agent = create_reflection_agent(model, inference_mode, input_message, role=CONTEXT_TEMPLATE)
    reflection_agent.run(
        input_data={"request": REQUEST, "context": CONTEXT},
    )
    assert reflection_agent._prompt.messages[1] == expected_result
    assert CONTEXT in reflection_agent._prompt.messages[0].content


@pytest.mark.parametrize(
    ("inference_mode"),
    [
        (InferenceMode.DEFAULT),
        (InferenceMode.XML),
        (InferenceMode.STRUCTURED_OUTPUT),
        (InferenceMode.FUNCTION_CALLING),
    ],
)
def test_custom_vision_agent_workflow(model, inference_mode):
    input_message = VisionMessage(
        content=[
            VisionMessageImageContent(image_url=VisionMessageImageURL(url=URL_TEMPLATE)),
            VisionMessageTextContent(text=REQUEST_TEMPLATE),
        ],
        role=MessageRole.USER,
    )

    agent = create_react_agent(model, inference_mode, input_message)

    agent.run(
        input_data={"request": REQUEST, "url": URL},
    )
    expected_result = input_message.format_message(request=REQUEST, url=URL)
    assert agent._prompt.messages[1] == expected_result

    simple_agent = create_simple_agent(model, inference_mode, input_message)
    simple_agent.run(
        input_data={"request": REQUEST, "url": URL},
    )
    assert simple_agent._prompt.messages[1] == expected_result

    reflection_agent = create_reflection_agent(model, inference_mode, input_message)
    reflection_agent.run(
        input_data={"request": REQUEST, "url": URL},
    )
    assert reflection_agent._prompt.messages[1] == expected_result
