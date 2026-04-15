import pytest

pytest.importorskip("pydantic_monty")

from dynamiq.nodes.agents.exceptions import ToolExecutionException  # noqa: E402
from dynamiq.nodes.tools.python_monty import PythonMonty  # noqa: E402


def test_basic_run_returns_dict():
    code = """
def run(inputs):
    return {"result": inputs["a"] + inputs["b"]}
"""
    node = PythonMonty(code=code)
    output = node.execute({"a": 2, "b": 3})
    assert output == {"content": {"result": 5}}


def test_use_multiple_params_true_unpacks_kwargs():
    code = """
def run(a, b):
    return {"sum": a + b}
"""
    node = PythonMonty(code=code, use_multiple_params=True)
    output = node.execute({"a": 10, "b": 32})
    assert output == {"content": {"sum": 42}}


def test_is_optimized_for_agents_stringifies_content():
    code = """
def run(inputs):
    return {"content": {"n": 7}}
"""
    node = PythonMonty(code=code, is_optimized_for_agents=True)
    output = node.execute({})
    assert output == {"content": str({"n": 7})}


def test_non_dict_return_wrapped_in_content():
    code = """
def run(inputs):
    return 42
"""
    node = PythonMonty(code=code)
    output = node.execute({})
    assert output == {"content": 42}


def test_dict_with_content_passthrough():
    code = """
def run(inputs):
    return {"content": {"nested": True}, "meta": "ok"}
"""
    node = PythonMonty(code=code, is_optimized_for_agents=False)
    output = node.execute({})
    assert output == {"content": {"nested": True}, "meta": "ok"}


def test_async_run_awaited():
    code = """
async def run(inputs):
    return {"doubled": inputs["x"] * 2}
"""
    node = PythonMonty(code=code)
    output = node.execute({"x": 21})
    assert output == {"content": {"doubled": 42}}


def test_missing_run_raises_tool_execution_exception():
    code = "x = 1\n"
    node = PythonMonty(code=code)
    with pytest.raises(ToolExecutionException) as excinfo:
        node.execute({})
    assert excinfo.value.recoverable is True
