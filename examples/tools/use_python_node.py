from pydantic import ConfigDict

from dynamiq.nodes.tools.python import Python


def run_python_node_with_input():
    """Basic example of running Python node with input data."""
    python_code_with_input = """
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

    python_node = Python(code=python_code_with_input, model_config=ConfigDict())
    input_data = {"name": "Alice", "age": 30}
    config = None
    result = python_node.run(input_data, config)
    print("Result with input:")
    print(result)


def run_python_node_without_input():
    """Basic example of running Python node without input data."""
    python_code_without_input = """
def run(input_data):
    # Generate a "random" number without using imports
    pseudo_random = hash(str(input_data)) % 100 + 1
    return {
        'pseudo_random_number': pseudo_random,
        'message': f'The generated number is {pseudo_random}.'
    }
"""

    python_node = Python(code=python_code_without_input, model_config=ConfigDict())
    result = python_node.run({})
    print("\nResult without input:")
    print(result)


def run_python_node_with_math_import():
    """Basic example of running Python node with math import."""
    python_code_with_math = """
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

    python_node = Python(code=python_code_with_math, model_config=ConfigDict())
    input_data = {"radius": 5}
    config = None
    result = python_node.run(input_data, config)
    print("\nResult with math import:")
    print(result)


def run_python_node_with_random_import():
    """Basic example of running Python node with random import."""
    python_code_with_random = """
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

    python_node = Python(code=python_code_with_random, model_config=ConfigDict())
    input_data = {"min": 1, "max": 10}
    config = None
    result = python_node.run(input_data, config)
    print("\nResult with random import:")
    print(result)


def run_python_node_with_dynamiq_import():
    """Basic example of running Python node with dynamiq relative import."""
    python_code_with_math = """

def run(input_data):
    from dynamiq.types import Document

    content = input_data.get('content')
    metadata = input_data.get('metadata', {})

    document = Document(content=content, metadata=metadata)
    return {
        'documents': [document,]
    }
"""

    python_node = Python(code=python_code_with_math, model_config=ConfigDict())

    input_data = {
        "content": "Document content",
        "metadata": {
            "title": "Document title",
            "author": "Document author",
        },
    }
    config = None
    result = python_node.run(input_data, config)
    print("\nResult with dynamiq relative import:")
    print(result)


if __name__ == "__main__":
    # run_python_node_with_input()
    # run_python_node_without_input()
    # run_python_node_with_math_import()
    # run_python_node_with_random_import()
    run_python_node_with_dynamiq_import()
