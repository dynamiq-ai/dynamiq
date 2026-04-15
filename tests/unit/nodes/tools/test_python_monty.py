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
