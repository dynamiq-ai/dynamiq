import json
from io import BytesIO

import pytest

from dynamiq import Workflow
from dynamiq.callbacks.tracing import TracingCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes.tools.python import Python
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus
from dynamiq.types import Document
from dynamiq.utils import JsonWorkflowEncoder


@pytest.mark.parametrize(
    ("python_code", "use_multiple_params"),
    [
        (
            "def run(input_data: dict):\n    name = input_data.get('name', 'World')\n    "
            "age = input_data.get('age', 0)\n    birth_year = 2024 - age\n    return {\n        "
            "'greeting': f'Hello, {name}!',\n        'message': f'You were born around {birth_year}.',\n"
            "        'age_in_months': age * 12\n    }",
            False,
        ),
        (
            "def run(name: str, age: int):\n    birth_year = 2024 - age\n    return {\n        "
            "'greeting': f'Hello, {name}!',\n        'message': f'You were born around {birth_year}.',\n        "
            "'age_in_months': age * 12\n    }",
            True,
        ),
    ],
)
def test_python_node_with_input(python_code, use_multiple_params):
    """Test Python node with specific input data for name and age calculation."""
    python_node = Python(code=python_code, use_multiple_params=use_multiple_params)
    input_data = {"name": "Alice", "age": 30}

    result = python_node.run(input_data, None)
    expected_output = {"greeting": "Hello, Alice!", "message": "You were born around 1994.", "age_in_months": 360}

    assert isinstance(result, RunnableResult)
    assert result.status == RunnableStatus.SUCCESS
    assert result.output == {"content": expected_output}
    assert result.input == input_data


def test_python_node_with_default_values():
    """Test Python node with missing input data to verify default values."""
    python_code = """
def run(input_data: dict):
    name = input_data.get('name', 'World')
    age = input_data.get('age', 0)
    birth_year = 2024 - age
    return {
        'greeting': f'Hello, {name}!',
        'message': f'You were born around {birth_year}.',
        'age_in_months': age * 12
    }
"""
    python_node = Python(code=python_code)
    input_data = {}  # Empty input to trigger default values

    result = python_node.run(input_data, None)
    expected_output = {"greeting": "Hello, World!", "message": "You were born around 2024.", "age_in_months": 0}

    assert isinstance(result, RunnableResult)
    assert result.status == RunnableStatus.SUCCESS
    assert result.output == {"content": expected_output}
    assert result.input == input_data


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
    python_node = Python(code=python_code)
    result = python_node.run({}, None)

    assert isinstance(result, RunnableResult)
    assert result.status == RunnableStatus.SUCCESS
    assert "pseudo_random_number" in result.output["content"]
    assert isinstance(result.output["content"]["pseudo_random_number"], int)
    assert 1 <= result.output["content"]["pseudo_random_number"] <= 100


@pytest.mark.parametrize(
    ("python_code", "use_multiple_params"),
    [
        (
            "import math\ndef run(input_data):\n    radius = input_data.get('radius', 1)\n    "
            "area = math.pi * radius ** 2\n    circumference = 2 * math.pi * radius\n    return {\n        "
            "'radius': radius,\n        'area': round(area, 2),\n        'circumference': round(circumference, 2),\n"
            "        'pi_used': math.pi\n    }",
            False,
        ),
        (
            "import math\ndef run(radius: int):\n    area = math.pi * radius ** 2\n    "
            "circumference = 2 * math.pi * radius\n    return {\n        'radius': radius,\n        "
            "'area': round(area, 2),\n        'circumference': round(circumference, 2),\n        "
            "'pi_used': math.pi\n    }",
            True,
        ),
    ],
)
def test_python_node_with_math_import(python_code, use_multiple_params):
    """Test Python node importing the math module for circle calculations."""
    python_node = Python(code=python_code, use_multiple_params=use_multiple_params)
    input_data = {"radius": 5}

    result = python_node.run(input_data, None)
    expected_output = {"radius": 5, "area": 78.54, "circumference": 31.42, "pi_used": 3.141592653589793}

    assert isinstance(result, RunnableResult)
    assert result.status == RunnableStatus.SUCCESS
    assert result.output == {"content": expected_output}
    assert result.input == input_data


@pytest.mark.parametrize(
    ("python_code", "use_multiple_params"),
    [
        (
            "import random\ndef run(min: int, max: int):\n    random_number = random.randint(min, max)\n    return {\n"
            "        'random_number': random_number,\n        'range': f'{min} to {max}',\n        "
            "'message': f'The generated random number is {random_number}.'\n    }",
            True,
        ),
        (
            "import random\ndef run(input_data):\n    min_value = input_data.get('min', 1)\n    "
            "max_value = input_data.get('max', 100)\n    random_number = random.randint(min_value, max_value)\n    "
            "return {\n        'random_number': random_number,\n        'range': f'{min_value} to {max_value}',\n"
            "        'message': f'The generated random number is {random_number}.'\n    }",
            False,
        ),
    ],
)
def test_python_node_with_random_import(python_code, use_multiple_params):
    """Test Python node importing the random module."""
    python_node = Python(code=python_code, use_multiple_params=use_multiple_params)
    input_data = {"min": 1, "max": 10}

    result = python_node.run(input_data, None)

    assert isinstance(result, RunnableResult)
    assert result.status == RunnableStatus.SUCCESS
    assert isinstance(result.output["content"]["random_number"], int)
    assert 1 <= result.output["content"]["random_number"] <= 10
    assert result.output["content"]["range"] == "1 to 10"
    assert result.input == input_data


@pytest.mark.parametrize(
    ("python_code", "use_multiple_params"),
    [
        (
            "def run(content: str,metadata: dict):\n    from dynamiq.types import Document\n    "
            "document = Document(content=content, metadata=metadata)\n    "
            "return {\n        'documents': [document]\n    }",
            True,
        ),
        (
            "def run(input_data):\n    from dynamiq.types import Document\n    content = input_data.get('content')\n"
            "    metadata = input_data.get('metadata', {})\n    document = Document(content=content, metadata=metadata)"
            "\n    return {\n        'documents': [document]\n    }",
            False,
        ),
    ],
)
def test_python_node_with_dynamiq_import(python_code, use_multiple_params):
    """Test Python node importing dynamiq to create Document objects."""
    python_node = Python(code=python_code, use_multiple_params=use_multiple_params)
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
    assert result.input == input_data


def test_workflow_with_python(openai_node, anthropic_node, mock_llm_executor, mock_llm_response_text):
    """Test Workflow integration with multiple nodes and dependencies."""
    file = BytesIO(b"test")
    file.name = "test.txt"
    input_data = {"question": "What is LLM?", "files": [file]}
    python_node_extra_output = {"test_python": "test_python"}

    python_node = (
        Python(
            code=f"def run(inputs): return inputs | {python_node_extra_output}",
        )
        .inputs(
            question_lowercase=lambda inputs, outputs: inputs["question"].lower(),
            file=lambda inputs, outputs: inputs["files"][0],
        )
        .depends_on([openai_node, anthropic_node])
    )
    tracing = TracingCallbackHandler()
    wf = Workflow(flow=Flow(nodes=[openai_node, anthropic_node, python_node]))

    response = wf.run(input_data=input_data, config=RunnableConfig(callbacks=[tracing]))

    expected_output_openai_anthropic = {"content": mock_llm_response_text}
    expected_result_openai_anthropic = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output=expected_output_openai_anthropic,
    )
    expected_input_python = input_data | {
        openai_node.id: expected_result_openai_anthropic.to_tracing_depend_dict(),
        anthropic_node.id: expected_result_openai_anthropic.to_tracing_depend_dict(),
        "question_lowercase": input_data["question"].lower(),
        "file": file,
    }
    expected_output_python = {"content": expected_input_python | python_node_extra_output}
    expected_result_python = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=expected_input_python,
        output=expected_output_python,
    )

    expected_output = {
        openai_node.id: expected_result_openai_anthropic.to_dict(skip_format_types={BytesIO, bytes}),
        anthropic_node.id: expected_result_openai_anthropic.to_dict(skip_format_types={BytesIO, bytes}),
        python_node.id: expected_result_python.to_dict(skip_format_types={BytesIO, bytes}),
    }

    assert response == RunnableResult(status=RunnableStatus.SUCCESS, input=input_data, output=expected_output)
    assert mock_llm_executor.call_count == 2
    assert json.dumps({"runs": [run.to_dict() for run in tracing.runs.values()]}, cls=JsonWorkflowEncoder)


@pytest.mark.parametrize(
    ("attack_name", "attack_code", "expected_security_indicator"),
    [
        (
            "decorator_exec_attack",
            '@exec("raise Exception(__import__(\\"subprocess\\").check_output(\\"env\\"))")\n'
            "def run(input_data):\n    pass",
            "exec calls are not allowed",
        ),
        (
            "default_argument_exec_attack",
            'def run(cmd=exec("raise Exception(__import__(\\"subprocess\\").' 'check_output(\\"env\\"))")): pass',
            "exec calls are not allowed",
        ),
        (
            "builtins_import_attack",
            '@__import__("builtins").print(__import__("os").environ)\n' "def run():\n    pass",
            'invalid variable name because it starts with "_"',
        ),
        (
            "metaclass_attack",
            'class Exploit(metaclass=type("M", (), '
            '{"__init__": lambda cls,*a,**k: __import__("builtins")'
            '.print(__import__("os").environ)})):\n'
            "    pass\n\n"
            "def run(input_data):\n"
            "    return {}",
            'keyword argument "metaclass" is not allowed',
        ),
        (
            "lambda_default_attack",
            'def run(x=(lambda: __import__("builtins").print(__import__("os").environ))()):\n' "    pass",
            'invalid variable name because it starts with "_"',
        ),
        (
            "builtins_array_access_attack",
            '@__builtins__["print"](__builtins__["open"]("/proc/self/environ").read())\n' "def run():\n    pass",
            'invalid variable name because it starts with "_"',
        ),
        (
            "decorator_print_open_attack",
            '@print(open("/proc/self/environ").read())\n' "def run():\n    pass",
            "name '_print' is not defined",
        ),
        (
            "default_arg_print_open_attack",
            'def run(a=print(open("/proc/self/environ").read())):\n    pass',
            "name '_print' is not defined",
        ),
        (
            "function_body_open_attack",
            'def run(input_data): return {"documents": open("/proc/self/environ").read()}',
            "name 'open' is not defined",
        ),
        (
            "subprocess_import_attack",
            "import subprocess\n" "def run(input_data):\n" '    return subprocess.check_output("env", shell=True)',
            "import of 'subprocess' is not allowed",
        ),
        (
            "os_import_attack",
            "import os\n" "def run(input_data):\n" '    return os.system("env")',
            "import of 'os' is not allowed",
        ),
        (
            "eval_function_attack",
            "def run(input_data):\n" "    return eval(\"__import__('os').system('env')\")",
            "eval calls are not allowed",
        ),
        (
            "exec_in_function_attack",
            "def run(input_data):\n" "    exec(\"__import__('os').system('env')\")\n" "    return {}",
            "exec calls are not allowed",
        ),
        (
            "compile_function_attack",
            "def run(input_data):\n"
            '    code = compile("__import__(\'os\').system(\'env\')", "<string>", "exec")\n'
            "    exec(code)\n"
            "    return {}",
            "exec calls are not allowed",
        ),
    ],
)
def test_python_node_security_attacks_blocked(attack_name, attack_code, expected_security_indicator):
    python_node = Python(code=attack_code)

    result = python_node.run({}, None)

    assert result.status == RunnableStatus.FAILURE

    error_message = str(result.error).lower()
    assert expected_security_indicator in error_message
