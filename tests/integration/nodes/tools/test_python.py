import collections
import copy
import datetime
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
    python_node = Python(
        code=python_code, use_multiple_params=use_multiple_params, extra_allowed_modules=["dynamiq"]
    )
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


@pytest.mark.parametrize(
    ("attack_name", "attack_code", "extra_allowed_modules", "expected_security_indicator"),
    [
        (
            # The reported escape: numpy re-exports the ctypes module, which grants native calls such as
            # system(). It must stay blocked even when numpy is explicitly enabled.
            "numpy_ctypes_native_call",
            "def run(input_data):\n    import numpy\n    return numpy.ctypeslib.ctypes.CDLL(None).getpid()\n",
            ["numpy"],
            "access to module 'ctypes'",
        ),
        (
            # urllib.request re-exports the os module, a path to os.environ that needs no import of os.
            "urllib_os_attribute",
            "def run(input_data):\n    import urllib.request\n    return urllib.request.os.environ\n",
            ["urllib"],
            "access to module 'os'",
        ),
        (
            # "from pkg.sub import name" resolves the name with a raw attribute read the getattr guard
            # never sees, so the fromlist entries are checked in restricted_import instead.
            "from_import_forbidden_module",
            "from numpy.ctypeslib import ctypes\ndef run(input_data):\n    return str(ctypes.CDLL)\n",
            ["numpy"],
            "not allowed",
        ),
        (
            "from_import_os_via_urllib",
            "from urllib.request import os\ndef run(input_data):\n    return os.getpid()\n",
            ["urllib"],
            "not allowed",
        ),
        (
            # from io import open binds the builtin open directly as a name, skipping the attribute guard.
            "from_import_builtin_open",
            "from io import open\ndef run(input_data):\n    return open('/etc/hostname').read()\n",
            [],
            "not allowed",
        ),
    ],
)
def test_python_node_module_traversal_blocked(
    attack_name, attack_code, extra_allowed_modules, expected_security_indicator
):
    """Reaching a disallowed module through an attribute of an allowed one is refused."""
    python_node = Python(code=attack_code, extra_allowed_modules=extra_allowed_modules)

    result = python_node.run({}, None)

    assert result.status == RunnableStatus.FAILURE
    assert expected_security_indicator in str(result.error).lower()


def test_python_node_rejects_extended_library_by_default():
    """numpy is not importable unless the node opts into it through extra_allowed_modules."""
    python_node = Python(code="def run(input_data):\n    import numpy\n    return 1\n")

    result = python_node.run({}, None)

    assert result.status == RunnableStatus.FAILURE
    assert "import of 'numpy' is not allowed" in str(result.error).lower()


def test_python_node_requests_and_typing_are_default():
    """requests and typing are importable without opting in; typing's safe members and a benign
    requests.utils helper still resolve."""
    code = (
        "import requests\n"
        "import typing\n"
        "from typing import Optional\n"
        "def run(input_data):\n"
        "    return {\n"
        "        'has_get': callable(requests.get),\n"
        "        'quote': requests.utils.quote('a b'),\n"
        "        'origin': str(typing.get_origin(Optional[int])),\n"
        "    }\n"
    )

    result = Python(code=code).run({}, None)

    assert result.status == RunnableStatus.SUCCESS
    assert result.output["content"]["has_get"] is True
    assert result.output["content"]["quote"] == "a%20b"
    assert "Union" in result.output["content"]["origin"]


def test_python_node_str_format_safe_templates_work():
    """str.format/format_map with plain fields and format specs is allowed; only fields that resolve
    attributes or items are refused."""
    code = (
        "def run(input_data):\n"
        "    return {\n"
        "        'positional': '{} {}'.format('a', 'b'),\n"
        "        'indexed': '{0}-{1}'.format(1, 2),\n"
        "        'named': '{name}'.format(name='x'),\n"
        "        'spec': '{:.2f}'.format(3.14159),\n"
        "        'nested_spec': '{0:>{1}}'.format('z', 5),\n"
        "        'format_map': '{a}'.format_map({'a': 7}),\n"
        "    }\n"
    )

    result = Python(code=code).run({}, None)

    assert result.status == RunnableStatus.SUCCESS
    content = result.output["content"]
    assert content["positional"] == "a b"
    assert content["indexed"] == "1-2"
    assert content["named"] == "x"
    assert content["spec"] == "3.14"
    assert content["nested_spec"] == "    z"
    assert content["format_map"] == "7"


@pytest.mark.parametrize(
    ("attack_name", "attack_code"),
    [
        # A replacement field that reaches a dunder through a nested format spec resolves in C.
        (
            "str_format_nested_spec_attr",
            "import uuid\ndef run(input_data):\n    return '{0:{1.__globals__}}'.format('x', uuid.uuid4)\n",
        ),
        # format_map resolves the same attribute walk through its field names.
        (
            "str_format_map_attr",
            "import uuid\ndef run(input_data):\n    return '{a.__globals__}'.format_map({'a': uuid.uuid4})\n",
        ),
    ],
)
def test_python_node_str_format_attribute_fields_still_blocked(attack_name, attack_code):
    """The str.format wrapper refuses attribute/index access even when hidden in a nested spec or reached
    through format_map."""
    result = Python(code=attack_code).run({}, None)

    assert result.status == RunnableStatus.FAILURE


def test_python_node_urllib_parse_request_and_error_are_default():
    """urllib.parse, urllib.request.Request and the urllib.error classes are importable by default, on
    both the attribute and from-import paths."""
    code = (
        "import urllib.parse\n"
        "from urllib.request import Request\n"
        "from urllib.error import HTTPError, URLError\n"
        "def run(input_data):\n"
        "    return {\n"
        "        'quote': urllib.parse.quote('a b/c'),\n"
        "        'encoded': urllib.parse.urlencode({'q': 'a b'}),\n"
        "        'scheme': Request('https://example.com').type,\n"
        "        'error_subclass': issubclass(HTTPError, URLError),\n"
        "    }\n"
    )

    result = Python(code=code).run({}, None)

    assert result.status == RunnableStatus.SUCCESS
    content = result.output["content"]
    assert content["quote"] == "a%20b/c"
    assert content["encoded"] == "q=a+b"
    assert content["scheme"] == "https"
    assert content["error_subclass"] is True


@pytest.mark.parametrize(
    ("attack_name", "attack_code"),
    [
        # urlopen('file://...') reads a local file past the open/io guards; refused on the attribute path,
        (
            "urlopen_file_attribute",
            "import urllib.request\n"
            "def run(input_data):\n    return urllib.request.urlopen('file:///etc/hostname').read()\n",
        ),
        # on the from-import path (the bound urlopen is the wrapped one too),
        (
            "urlopen_file_from_import",
            "from urllib.request import urlopen\n"
            "def run(input_data):\n    return urlopen('file:///etc/hostname').read()\n",
        ),
        # and when the scheme is carried on a Request object.
        (
            "urlopen_file_request",
            "import urllib.request\n"
            "def run(input_data):\n"
            "    return urllib.request.urlopen(urllib.request.Request('file:///etc/hostname')).read()\n",
        ),
        # The file-reading/writing helpers are refused outright, by attribute and by from-import.
        (
            "urlretrieve_attribute",
            "import urllib.request\n"
            "def run(input_data):\n    return urllib.request.urlretrieve('http://x', '/tmp/x')\n",
        ),
        (
            "urlretrieve_from_import",
            "from urllib.request import urlretrieve\ndef run(input_data):\n    return urlretrieve\n",
        ),
        ("urlopener", "import urllib.request\ndef run(input_data):\n    return urllib.request.URLopener()\n"),
    ],
)
def test_python_node_urllib_file_access_blocked(attack_name, attack_code):
    """With urllib in the default set, urlopen is restricted to http(s) (no file://) and the file helpers
    are refused, so urllib grants no filesystem access."""
    result = Python(code=attack_code).run({}, None)

    assert result.status == RunnableStatus.FAILURE


def test_safe_urlopen_allows_http_and_blocks_other_schemes(monkeypatch):
    """The urlopen wrapper lets http(s) through to the real urlopen and refuses every other scheme,
    without a network call."""
    from dynamiq.nodes.tools import python as python_module

    calls = []
    monkeypatch.setattr(python_module, "_URLLIB_URLOPEN", lambda url, *a, **k: calls.append(url) or "RESPONSE")

    assert python_module._safe_urlopen("https://example.com/path") == "RESPONSE"
    assert calls == ["https://example.com/path"]
    for blocked_url in ("file:///etc/hostname", "ftp://host/f", "data:text/plain,hi"):
        with pytest.raises(ValueError):
            python_module._safe_urlopen(blocked_url)


@pytest.mark.parametrize(
    ("attack_name", "attack_code"),
    [
        # requests.utils.atomic_open opens an arbitrary path for writing, past the open/io guards.
        (
            "requests_atomic_open",
            "import requests\ndef run(input_data):\n    return requests.utils.atomic_open('/tmp/pwn')\n",
        ),
        # get_environ_proxies and compat.getproxies return the process proxy environment.
        (
            "requests_get_environ_proxies",
            "import requests\ndef run(input_data):\n    return requests.utils.get_environ_proxies('https://x')\n",
        ),
        (
            "requests_compat_getproxies",
            "import requests\ndef run(input_data):\n    return requests.compat.getproxies()\n",
        ),
        # get_netrc_auth returns the login/password from ~/.netrc.
        (
            "requests_get_netrc_auth",
            "import requests\ndef run(input_data):\n    return requests.utils.get_netrc_auth('https://x')\n",
        ),
    ],
)
def test_python_node_requests_file_and_credential_helpers_blocked(attack_name, attack_code):
    """requests stays in the default, but its file-write and credential helpers are refused by identity.

    This closes the reachable helpers only; the wider requests/urllib3 surface and trust_env are not a
    boundary for untrusted code, which belongs in an out-of-process sandbox node.
    """
    result = Python(code=attack_code).run({}, None)

    assert result.status == RunnableStatus.FAILURE


def test_python_node_extra_allowed_modules_enables_library():
    """An opted-in library works, and legitimate submodule access (numpy.linalg) still resolves."""
    code = "def run(input_data):\n    import numpy\n    return {'norm': float(numpy.linalg.norm([3.0, 4.0]))}\n"
    python_node = Python(code=code, extra_allowed_modules=["numpy"])

    result = python_node.run({}, None)

    assert result.status == RunnableStatus.SUCCESS
    assert result.output["content"]["norm"] == 5.0


def test_python_node_forbidden_module_cannot_be_opted_in():
    """A forbidden module stays blocked even if a caller lists it in extra_allowed_modules."""
    python_node = Python(
        code="def run(input_data):\n    import os\n    return 1\n", extra_allowed_modules=["os"]
    )

    result = python_node.run({}, None)

    assert result.status == RunnableStatus.FAILURE
    assert "import of 'os' is not allowed" in str(result.error).lower()


@pytest.mark.parametrize(
    ("attack_name", "attack_code"),
    [
        # vars() hands back a module's or class's raw __dict__, bypassing the attribute guard.
        ("vars_module_dict", "import uuid\ndef run(input_data):\n    return vars(uuid)['os'].getpid()\n"),
        (
            "vars_type_subclasses",
            "def run(input_data):\n    return vars(type)['__subclasses__'](object)\n",
        ),
        # io.open is the builtin open: a filesystem primitive reached through an allowed module.
        ("io_open_file", "import io\ndef run(input_data):\n    return io.open('/etc/hostname').read()\n"),
        # Mutating a module or a class an allowed module defines leaks state into every other run.
        ("setattr_module", "import json\ndef run(input_data):\n    setattr(json, 'x', 1)\n    return 1\n"),
        (
            "write_shared_class_attr",
            "import json\ndef run(input_data):\n    json.JSONEncoder.default = 1\n    return 1\n",
        ),
    ],
)
def test_python_node_object_and_write_escapes_blocked(attack_name, attack_code):
    """Object-leak and shared-state-write paths that do not go through an import are refused."""
    result = Python(code=attack_code).run({}, None)

    assert result.status == RunnableStatus.FAILURE


@pytest.mark.parametrize(
    ("attack_name", "attack_code"),
    [
        # getattr with a string name skips the compiler's underscore check; the guarded builtin must not.
        (
            "getattr_subclasses_jailbreak",
            "def run(input_data):\n"
            "    subclasses = getattr(object, '__subclasses__')()\n"
            "    wrapper = [c for c in subclasses if c.__name__ == '_wrap_close'][0]\n"
            "    return getattr(getattr(wrapper, '__init__'), '__globals__')['system']('id')\n",
        ),
        (
            "getattr_string_globals",
            "import json\ndef run(input_data):\n    return getattr(json.dumps, '__globals__')\n",
        ),
        # io.FileIO and io.open_code open files like io.open, which the identity guard also refuses.
        ("io_fileio_read", "import io\ndef run(input_data):\n    return io.FileIO('/etc/hostname').read()\n"),
        ("io_open_code_read", "import io\ndef run(input_data):\n    return io.open_code('/etc/hostname').read()\n"),
    ],
)
def test_python_node_getattr_and_io_escapes_blocked(attack_name, attack_code):
    """String-name getattr and the io file openers cannot escape the default allow-list."""
    result = Python(code=attack_code).run({}, None)

    assert result.status == RunnableStatus.FAILURE


def test_python_node_operator_not_in_default_allow_list():
    """operator is excluded from the default: attrgetter/methodcaller call C getattr past every guard."""
    result = Python(code="import operator\ndef run(input_data):\n    return 1\n").run({}, None)

    assert result.status == RunnableStatus.FAILURE
    assert "import of 'operator' is not allowed" in str(result.error).lower()


@pytest.mark.parametrize(
    ("attack_name", "attack_code"),
    [
        # str.format replacement fields resolve attributes in C, past every guard.
        (
            "str_format_dunder_walk",
            "import uuid\ndef run(input_data):\n    return str('{0.__globals__[os]}'.format(uuid.uuid4))\n",
        ),
        # string and typing are no longer in the default set: string.Formatter walks dunder fields and
        # typing.get_type_hints evals string annotations, both outside the guards.
        ("string_not_in_default", "import string\ndef run(input_data):\n    return 1\n"),
        # typing is allowed by default, but its eval entry points are refused (see the dedicated test below).
        (
            "typing_get_type_hints_eval",
            "import typing\n"
            "def f(x: '().__class__.__bases__[0].__subclasses__()'):\n    pass\n"
            "def run(input_data):\n    return len(typing.get_type_hints(f)['x'])\n",
        ),
        ("typing_forward_ref", "import typing\ndef run(input_data):\n    return typing.ForwardRef('1+1')\n"),
        # A ForwardRef instance is reachable without the class via get_args; on 3.14 its evaluate() evals.
        (
            "typing_forwardref_instance",
            "import typing\ndef run(input_data):\n    return typing.get_args(typing.List['1+1'])[0]\n",
        ),
        (
            "typing_forwardref_evaluate_backstop",
            "import typing\n"
            "def run(input_data):\n"
            "    (ref,) = typing.get_args(typing.List['1+1'])\n"
            "    return ref.evaluate()\n",
        ),
        # setattr on a shared library function would change it for every run; the dunder name is blocked too.
        (
            "setattr_shared_function",
            "import json\n"
            "def run(input_data):\n"
            "    setattr(json.dumps, '__kwdefaults__', {'indent': 7})\n"
            "    return 1\n",
        ),
        # collections.UserString.format delegates to str.format in library code, past the str.format guard.
        (
            "userstring_format",
            "import collections, uuid\n"
            "def run(input_data):\n"
            "    return str(collections.UserString('{0.__globals__[os]}').format(uuid.uuid4))\n",
        ),
        # A str subclass or a super() proxy forwards to the real str.format; the value-based check catches it.
        (
            "str_subclass_format",
            "class S(str):\n    pass\ndef run(input_data):\n    return str(S('{0.__globals__[os]}').format(1))\n",
        ),
        (
            "super_proxy_format",
            "import uuid\n"
            "class S(str):\n    pass\n"
            "def run(input_data):\n"
            "    return str(super(S, S('{0.__globals__[os]}')).format(uuid.uuid4))\n",
        ),
        (
            "str_format_unbound",
            "def run(input_data):\n    return str(str.format('{0.__globals__[os]}', run))\n",
        ),
        # functools.singledispatch(...).register runs typing.get_type_hints on the function, evaluating
        # its string annotations in library code.
        (
            "functools_singledispatch_register",
            "import functools\n"
            "@functools.singledispatch\n"
            "def g(x):\n    return 0\n"
            "def h(x: '().__class__.__bases__[0].__subclasses__()'):\n    return 1\n"
            "def run(input_data):\n    g.register(h)\n    return 1\n",
        ),
        # functools.update_wrapper / wraps run caller-named getattr/setattr in library code: they can copy
        # __globals__ into the snippet or overwrite a shared function's __code__.
        (
            "functools_update_wrapper",
            "import functools\n"
            "class H:\n    pass\n"
            "def run(input_data):\n"
            "    def f():\n        pass\n"
            "    functools.update_wrapper(H(), f, assigned=('__globals__',), updated=())\n"
            "    return 1\n",
        ),
    ],
)
def test_python_node_c_level_escapes_blocked(attack_name, attack_code):
    """Stdlib functions that resolve attributes or eval in C cannot escape the default allow-list."""
    result = Python(code=attack_code).run({}, None)

    assert result.status == RunnableStatus.FAILURE


def test_python_node_ctypes_object_blocked_even_with_numpy():
    """A native handle from numpy.ctypeslib.load_library is refused by its ctypes type, on any path."""
    code = (
        "import numpy\n"
        "def run(input_data):\n"
        "    return numpy.ctypeslib.load_library('libc.so.6', '/lib/x86_64-linux-gnu').getpid()\n"
    )

    result = Python(code=code, extra_allowed_modules=["numpy"]).run({}, None)

    assert result.status == RunnableStatus.FAILURE


def test_python_node_getattr_allows_normal_and_whitelisted_names():
    """The hardened getattr builtin still serves ordinary names and the whitelisted dunders."""
    code = (
        "def run(input_data):\n"
        "    class Local:\n"
        "        pass\n"
        "    obj = Local()\n"
        "    setattr(obj, 'value', 7)\n"
        "    return {'value': getattr(obj, 'value'), 'cls': getattr(obj, '__class__').__name__}\n"
    )

    result = Python(code=code).run({}, None)

    assert result.status == RunnableStatus.SUCCESS
    assert result.output["content"] == {"value": 7, "cls": "Local"}


def test_python_node_allows_writes_to_run_local_objects():
    """Writing to objects the snippet creates stays allowed; only shared modules and classes are refused."""
    code = (
        "def run(input_data):\n"
        "    class Local:\n"
        "        pass\n"
        "    Local.attr = 1\n"
        "    instance = Local()\n"
        "    setattr(instance, 'value', 2)\n"
        "    return {'attr': Local.attr, 'value': instance.value}\n"
    )

    result = Python(code=code).run({}, None)

    assert result.status == RunnableStatus.SUCCESS
    assert result.output["content"] == {"attr": 1, "value": 2}


def test_python_node_allows_writes_to_snippet_defined_function():
    """A function the snippet defines is per-run, so attributes on it (counters, memo caches) are allowed."""
    code = (
        "def helper():\n"
        "    return 1\n"
        "def run(input_data):\n"
        "    helper.calls = 3\n"
        "    setattr(helper, 'label', 'x')\n"
        "    return {'calls': helper.calls, 'label': helper.label}\n"
    )

    result = Python(code=code).run({}, None)

    assert result.status == RunnableStatus.SUCCESS
    assert result.output["content"] == {"calls": 3, "label": "x"}


def test_python_node_refuses_writes_to_shared_library_function():
    """Writing to a function an allowed module defines is refused: it is shared across runs."""
    result = Python(
        code="import json\ndef run(input_data):\n    json.dumps.foo = 1\n    return 1\n",
        extra_allowed_modules=["json"],
    ).run({}, None)

    assert result.status == RunnableStatus.FAILURE


def test_python_node_refuses_writes_through_typing_alias():
    """A typing generic alias forwards attribute writes to its backing stdlib class, so the write is
    refused and collections.Counter is left intact for every other run."""
    code = (
        "import typing\n"
        "def evil(self, *args):\n"
        "    return 'hijacked'\n"
        "def run(input_data):\n"
        "    typing.Counter.most_common = evil\n"
        "    return 1\n"
    )
    original_most_common = collections.Counter.most_common
    try:
        result = Python(code=code).run({}, None)

        assert result.status == RunnableStatus.FAILURE
        assert collections.Counter("aab").most_common() == [("a", 2), ("b", 1)]
    finally:
        collections.Counter.most_common = original_most_common


def test_python_node_refuses_copy_dispatch_table_write():
    """copy.dispatch_table backs deepcopy and pickle for the whole process, so it is refused by identity
    and a run cannot install code that every later deepcopy of a type would call."""
    code = (
        "import copy, datetime\n"
        "def evil(obj):\n"
        "    return (str, ('hijacked',))\n"
        "def run(input_data):\n"
        "    copy.dispatch_table[datetime.date] = evil\n"
        "    return 1\n"
    )
    try:
        result = Python(code=code).run({}, None)

        assert result.status == RunnableStatus.FAILURE
        assert datetime.date not in copy.dispatch_table
        assert copy.deepcopy(datetime.date(2020, 1, 1)) == datetime.date(2020, 1, 1)
    finally:
        copy.dispatch_table.pop(datetime.date, None)


@pytest.mark.parametrize(
    ("attack_name", "attack_code"),
    [
        # "import a.b.c as x" binds the leaf through a raw IMPORT_FROM the getattr guard never sees, with
        # an empty fromlist; requests.packages.urllib3 is an alias for the real urllib3, off the allow-list.
        (
            "import_alias_urllib3",
            "import requests.packages.urllib3 as u\ndef run(input_data):\n    return u.__name__\n",
        ),
        (
            "import_alias_urllib3_connection",
            "import requests.packages.urllib3.util.connection as c\n"
            "def run(input_data):\n    return str(c.create_connection)\n",
        ),
    ],
)
def test_python_node_dotted_import_alias_to_forbidden_module_blocked(attack_name, attack_code):
    """`import a.b.c as x` cannot bind a submodule aliased to a module outside the allow-list."""
    result = Python(code=attack_code).run({}, None)

    assert result.status == RunnableStatus.FAILURE


def test_python_node_allows_dotted_submodule_import_within_allow_list():
    """A dotted import whose every component stays within an allowed package still works."""
    code = "import json.decoder as jd\ndef run(input_data):\n    return jd.__name__\n"

    result = Python(code=code).run({}, None)

    assert result.status == RunnableStatus.SUCCESS
    assert result.output["content"] == "json.decoder"


def test_python_node_from_import_aliased_submodule_blocked():
    """`from requests.packages.urllib3.util.connection import create_connection` loads an off-allow-list
    leaf module (the real urllib3) even with a non-empty fromlist; the dotted-prefix walk refuses it, so
    the snippet can't reach a raw socket factory."""
    code = (
        "from requests.packages.urllib3.util.connection import create_connection\n"
        "def run(input_data):\n    return str(create_connection)\n"
    )

    result = Python(code=code).run({}, None)

    assert result.status == RunnableStatus.FAILURE


def test_python_node_forwardref_class_attribute_access_blocked():
    """A ForwardRef reached as a class via type(f)/f.__class__ cannot expose evaluate(): attribute access
    on the ForwardRef class is refused, so the 3.14 eval entry point stays closed on the default config."""
    code = (
        "import typing\n"
        "def run(input_data):\n"
        "    for f in typing.get_args(typing.List['1+1']):\n"
        "        return str(type(f).evaluate)\n"
    )

    result = Python(code=code).run({}, None)

    assert result.status == RunnableStatus.FAILURE


def test_python_node_description_is_honest_about_imports_and_requests():
    """The agent-facing description must not claim the whole stdlib imports or present requests as opt-in,
    while a caller-supplied description (used as a tool label elsewhere) is still respected."""
    description = Python(code="def run(input_data):\n    return 1\n").description

    assert "requests is importable by default" in description
    assert "not the whole standard library" in description
    assert Python(code="def run(input_data):\n    return 1\n", description="calc").description == "calc"
