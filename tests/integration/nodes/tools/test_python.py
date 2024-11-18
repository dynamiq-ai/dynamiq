from pydantic import ConfigDict

from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.nodes.tools.python import Python
from dynamiq.runnables import RunnableResult, RunnableStatus
from dynamiq.types import Document


def test_python_node_with_input():
    """Test Python node with specific input data for name and age calculation."""
    python_code = """
def run(input_data):
    name = input_data.get('name', 'World')
    age = input_data.get('age', 0)
    birth_year = 2024 - age
    return {
        'greeting': f'Hello, {name}!',
        'message': f'You were born around {birth_year}.',
        'age_in_months': age * 12
    }
"""
    python_node = Python(code=python_code, model_config=ConfigDict())
    input_data = {"name": "Alice", "age": 30}

    result = python_node.run(input_data, None)
    expected_output = {"greeting": "Hello, Alice!", "message": "You were born around 1994.", "age_in_months": 360}

    assert isinstance(result, RunnableResult)
    assert result.status == RunnableStatus.SUCCESS
    assert result.output == {"content": expected_output}
    assert result.input.model_dump() == input_data


def test_python_node_with_default_values():
    """Test Python node with missing input data to verify default values."""
    python_code = """
def run(input_data):
    name = input_data.get('name', 'World')
    age = input_data.get('age', 0)
    birth_year = 2024 - age
    return {
        'greeting': f'Hello, {name}!',
        'message': f'You were born around {birth_year}.',
        'age_in_months': age * 12
    }
"""
    python_node = Python(code=python_code, model_config=ConfigDict())
    input_data = {}  # Empty input to trigger default values

    result = python_node.run(input_data, None)
    expected_output = {"greeting": "Hello, World!", "message": "You were born around 2024.", "age_in_months": 0}

    assert isinstance(result, RunnableResult)
    assert result.status == RunnableStatus.SUCCESS
    assert result.output == {"content": expected_output}
    assert result.input.model_dump() == input_data


def test_python_node_without_input():
    """Test Python node functionality without specific input data."""
    python_code = """
def run(input_data):
    pseudo_random = hash(str(input_data)) % 100 + 1
    return {
        'pseudo_random_number': pseudo_random,
        'message': f'The generated number is {pseudo_random}.'
    }
"""
    python_node = Python(code=python_code, model_config=ConfigDict())
    result = python_node.run({}, None)

    assert isinstance(result, RunnableResult)
    assert result.status == RunnableStatus.SUCCESS
    assert "pseudo_random_number" in result.output["content"]
    assert isinstance(result.output["content"]["pseudo_random_number"], int)
    assert 1 <= result.output["content"]["pseudo_random_number"] <= 100


def test_python_node_with_math_import():
    """Test Python node importing the math module for circle calculations."""
    python_code = """
import math

def run(input_data):
    radius = input_data.get('radius', 1)
    area = math.pi * radius ** 2
    circumference = 2 * math.pi * radius
    return {
        'radius': radius,
        'area': round(area, 2),
        'circumference': round(circumference, 2),
        'pi_used': math.pi
    }
"""
    python_node = Python(code=python_code, model_config=ConfigDict())
    input_data = {"radius": 5}

    result = python_node.run(input_data, None)
    expected_output = {"radius": 5, "area": 78.54, "circumference": 31.42, "pi_used": 3.141592653589793}

    assert isinstance(result, RunnableResult)
    assert result.status == RunnableStatus.SUCCESS
    assert result.output == {"content": expected_output}
    assert result.input.model_dump() == input_data


def test_python_node_with_random_import():
    """Test Python node importing the random module."""
    python_code = """
import random

def run(input_data):
    min_value = input_data.get('min', 1)
    max_value = input_data.get('max', 100)
    random_number = random.randint(min_value, max_value)
    return {
        'random_number': random_number,
        'range': f'{min_value} to {max_value}',
        'message': f'The generated random number is {random_number}.'
    }
"""
    python_node = Python(code=python_code, model_config=ConfigDict())
    input_data = {"min": 1, "max": 10}

    result = python_node.run(input_data, None)

    assert isinstance(result, RunnableResult)
    assert result.status == RunnableStatus.SUCCESS
    assert isinstance(result.output["content"]["random_number"], int)
    assert 1 <= result.output["content"]["random_number"] <= 10
    assert result.output["content"]["range"] == "1 to 10"
    assert result.input.model_dump() == input_data


def test_python_node_with_dynamiq_import():
    """Test Python node importing dynamiq to create Document objects."""
    python_code = """
def run(input_data):
    from dynamiq.types import Document
    content = input_data.get('content')
    metadata = input_data.get('metadata', {})
    document = Document(content=content, metadata=metadata)
    return {
        'documents': [document]
    }
"""
    python_node = Python(code=python_code, model_config=ConfigDict())
    input_data = {
        "content": "Document content",
        "metadata": {
            "title": "Document title",
            "author": "Document author",
        },
    }

    result = python_node.run(input_data, None)

    assert isinstance(result, RunnableResult)
    assert result.status == RunnableStatus.SUCCESS
    assert len(result.output["content"]["documents"]) == 1
    assert isinstance(result.output["content"]["documents"][0], Document)
    assert result.output["content"]["documents"][0].content == "Document content"
    assert result.output["content"]["documents"][0].metadata == input_data["metadata"]
    assert result.input.model_dump() == input_data


def test_workflow_with_python(openai_node, anthropic_node, mock_llm_executor, mock_llm_response_text):
    """Test Workflow integration with multiple nodes and dependencies."""
    input_data = {"question": "What is LLM?"}
    python_node_extra_output = {"test_python": "test_python"}

    python_node = (
        Python(
            code=f"def run(inputs): return inputs | {python_node_extra_output}",
        )
        .inputs(question_lowercase=lambda inputs, outputs: inputs["question"].lower())
        .depends_on([openai_node, anthropic_node])
    )
    wf = Workflow(flow=Flow(nodes=[openai_node, anthropic_node, python_node]))

    response = wf.run(input_data=input_data)

    expected_output_openai_anthropic = {"content": mock_llm_response_text, "tool_calls": None}
    expected_result_openai_anthropic = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output=expected_output_openai_anthropic,
    )
    expected_input_python = input_data | {
        openai_node.id: expected_result_openai_anthropic.to_tracing_depend_dict(),
        anthropic_node.id: expected_result_openai_anthropic.to_tracing_depend_dict(),
        "question_lowercase": input_data["question"].lower(),
    }
    expected_output_python = {"content": expected_input_python | python_node_extra_output}
    expected_result_python = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=expected_input_python,
        output=expected_output_python,
    )

    expected_output = {
        openai_node.id: expected_result_openai_anthropic.to_dict(),
        anthropic_node.id: expected_result_openai_anthropic.to_dict(),
        python_node.id: expected_result_python.to_dict(),
    }

    assert response == RunnableResult(status=RunnableStatus.SUCCESS, input=input_data, output=expected_output)
    assert mock_llm_executor.call_count == 2
