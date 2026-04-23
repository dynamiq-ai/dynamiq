import pytest

pytest.importorskip("pydantic_monty")

from dynamiq import Workflow  # noqa: E402
from dynamiq.flows import Flow  # noqa: E402
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


def test_disallowed_third_party_import_raises():
    code = """
import pandas
def run(inputs):
    return {"v": pandas.DataFrame().shape}
"""
    node = PythonMonty(code=code)
    with pytest.raises(ToolExecutionException) as excinfo:
        node.execute({})
    assert excinfo.value.recoverable is True
    assert "pandas" in str(excinfo.value)


def test_missing_pydantic_monty_raises_unrecoverable(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pydantic_monty":
            raise ImportError("simulated: pydantic_monty not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    node = PythonMonty(code="def run(inputs):\n    return 1\n")
    with pytest.raises(ToolExecutionException) as excinfo:
        node.execute({})
    assert excinfo.value.recoverable is False
    assert "pydantic-monty" in str(excinfo.value)


def test_unsupported_class_definition_raises():
    code = """
class Foo:
    pass
def run(inputs):
    return Foo()
"""
    node = PythonMonty(code=code)
    with pytest.raises(ToolExecutionException) as excinfo:
        node.execute({})
    assert excinfo.value.recoverable is True
    assert "class" in str(excinfo.value).lower()


@pytest.mark.asyncio
async def test_execute_async_directly():
    code = """
def run(inputs):
    return {"v": inputs["x"]}
"""
    node = PythonMonty(code=code)
    output = await node.execute_async({"x": 99})
    assert output == {"content": {"v": 99}}


def test_type_check_true_accepts_valid_code():
    code = """
def run(inputs: dict) -> dict:
    return {"ok": True}
"""
    node = PythonMonty(code=code, type_check=True)
    output = node.execute({})
    assert output == {"content": {"ok": True}}


def test_type_check_true_rejects_type_errors():
    code = """
def run(inputs: dict) -> int:
    return "not-an-int"
"""
    node = PythonMonty(code=code, type_check=True)
    with pytest.raises(ToolExecutionException) as excinfo:
        node.execute({})
    assert excinfo.value.recoverable is True


def test_python_monty_yaml_roundtrip(tmp_path):
    code = 'def run(inputs):\n    return {"value": inputs.get("x", 0) + 1}\n'
    node = PythonMonty(
        id="monty-node",
        code=code,
        use_multiple_params=False,
        type_check=False,
    )
    workflow = Workflow(id="monty-workflow", flow=Flow(id="monty-flow", nodes=[node]))

    yaml_path = tmp_path / "monty_workflow.yaml"
    workflow.to_yaml_file(str(yaml_path))

    loaded = Workflow.from_yaml_file(str(yaml_path), init_components=True)
    assert isinstance(loaded.flow.nodes[0], PythonMonty)
    loaded_node = loaded.flow.nodes[0]
    assert loaded_node.id == "monty-node"
    assert loaded_node.code == code
    assert loaded_node.use_multiple_params is False
    assert loaded_node.type_check is False

    roundtrip_path = tmp_path / "monty_workflow_roundtrip.yaml"
    loaded.to_yaml_file(str(roundtrip_path))
    roundtrip = Workflow.from_yaml_file(str(roundtrip_path), init_components=True)
    roundtrip_node = roundtrip.flow.nodes[0]
    assert roundtrip_node.id == "monty-node"
    assert roundtrip_node.code == code
    assert roundtrip_node.execute({"x": 4}) == {"content": {"value": 5}}
