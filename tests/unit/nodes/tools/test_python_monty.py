import datetime
import json
import pathlib
import subprocess
import sys
import textwrap
import time
import zoneinfo

import pytest

pytest.importorskip("pydantic_monty")

from dynamiq import Workflow  # noqa: E402
from dynamiq.flows import Flow  # noqa: E402
from dynamiq.nodes.agents.exceptions import ToolExecutionException  # noqa: E402
from dynamiq.nodes.node import ErrorHandling  # noqa: E402
from dynamiq.nodes.tools.python_monty import PythonMonty, _output_size  # noqa: E402
from dynamiq.utils.monty import MONTY_MAX_WORKERS  # noqa: E402
from dynamiq.utils.utils import JsonWorkflowEncoder, format_value  # noqa: E402


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


def test_missing_pydantic_monty_names_the_extra():
    # A fresh interpreter, so the hidden package is not already imported.
    script = textwrap.dedent(
        """
        import sys


        class HideMonty:
            def find_spec(self, name, path=None, target=None):
                if name == "pydantic_monty" or name.startswith("pydantic_monty."):
                    raise ModuleNotFoundError(f"No module named {name!r}", name=name)
                return None


        sys.meta_path.insert(0, HideMonty())
        from dynamiq.nodes import tools

        try:
            tools.PythonMonty
        except ModuleNotFoundError as error:
            print(type(error).__name__, error)
        """
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    assert result.stdout.startswith("OptionalDependencyError")
    assert 'pip install "dynamiq[monty]"' in result.stdout


def test_class_definition_runs():
    code = """
class Counter:
    def __init__(self, start):
        self.value = start

    def bump(self):
        self.value = self.value + 1
        return self.value

def run(inputs):
    return {"value": Counter(inputs["start"]).bump()}
"""
    node = PythonMonty(code=code)
    assert node.execute({"start": 41}) == {"content": {"value": 42}}


def test_endless_loop_stops_at_node_timeout():
    code = """
def run(inputs):
    while True:
        pass
"""
    node = PythonMonty(code=code, error_handling=ErrorHandling(timeout_seconds=1))
    started = time.monotonic()
    with pytest.raises(ToolExecutionException, match="time limit exceeded") as excinfo:
        node.execute({})
    assert time.monotonic() - started < 5
    assert excinfo.value.recoverable is True


def test_sleeps_return_at_once():
    # Monty's time limit does not count sleeping, so a real sleep would hold the worker past it.
    code = """
import asyncio
import time

async def run(inputs):
    for _ in range(1000):
        time.sleep(10)
        await asyncio.sleep(10)
    return {"slept": True}
"""
    node = PythonMonty(code=code, error_handling=ErrorHandling(timeout_seconds=1))
    started = time.monotonic()
    assert node.execute({}) == {"content": {"slept": True}}
    assert time.monotonic() - started < 2


def test_sleeping_runs_leave_workers_free_for_other_runs():
    sleeper = PythonMonty(
        code="import time\ndef run(inputs):\n    for _ in range(4):\n        time.sleep(10)\n    return 1\n",
        error_handling=ErrorHandling(timeout_seconds=0.5),
    )
    for _ in range(MONTY_MAX_WORKERS):
        sleeper.run(input_data={})
    started = time.monotonic()
    assert PythonMonty(code="def run(inputs):\n    return 2\n").execute({}) == {"content": 2}
    assert time.monotonic() - started < 1


# Each level references the previous list twice, so 22 levels of a few dozen objects print as ~29 million
# characters. The variants hide that from a size check that runs in the snippet's own namespace.
SHARED_SUBLISTS = """
def build():
    x = [0]
    for _ in range(22):
        x = [x, x]
    return x
"""
OVERSIZED_RESULTS = {
    "plain": SHARED_SUBLISTS + "def run(inputs):\n    return build()\n",
    "shadowed repr": SHARED_SUBLISTS + "def repr(value):\n    return ''\ndef run(inputs):\n    return build()\n",
    "shadowed len": SHARED_SUBLISTS + "def len(value):\n    return 0\ndef run(inputs):\n    return build()\n",
    "short __repr__": SHARED_SUBLISTS
    + """
class Result:
    def __init__(self, payload):
        self.payload = payload

    def __repr__(self):
        return "small"

def run(inputs):
    return Result(build())
""",
}
# A timezone prints its name, which can be any length, and an aware datetime or time prints its timezone.
LONG_TIMEZONE = "import datetime\ntz = datetime.timezone(datetime.timedelta(hours=1), 'x' * 100000)\n"
OVERSIZED_RESULTS |= {
    "timezone name": LONG_TIMEZONE + "def run(inputs):\n    return [tz] * 1000\n",
    "aware datetime": LONG_TIMEZONE
    + "def run(inputs):\n    return [datetime.datetime(2020, 1, 1, tzinfo=tz)] * 1000\n",
    "aware time": LONG_TIMEZONE + "def run(inputs):\n    return [datetime.time(1, tzinfo=tz)] * 1000\n",
    # Each frozenset({1}) prints 14 characters, so the container's own text has to count too.
    "frozensets": "def run(inputs):\n    return [frozenset({1})] * 800000\n",
}


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("code", OVERSIZED_RESULTS.values(), ids=OVERSIZED_RESULTS.keys())
async def test_oversized_result_stops_at_the_output_limit(code, use_async):
    node = PythonMonty(code=code, is_optimized_for_agents=True)
    started = time.monotonic()
    with pytest.raises(ToolExecutionException, match="characters of output"):
        if use_async:
            await node.execute_async({})
        else:
            node.execute({})
    assert time.monotonic() - started < 5


def renderings(value) -> dict[str, int]:
    """Return the length of each text the host makes of a result, skipping one that rejects the value."""
    renders = {
        "repr": repr,
        "str": str,
        "tracing JSON": lambda v: json.dumps(format_value(v, for_tracing=True), cls=JsonWorkflowEncoder),
        "agent JSON": lambda v: json.dumps(v, ensure_ascii=False, default=str),
        "JSON with str fallback": lambda v: json.dumps(v, default=str),
    }
    lengths = {}
    for name, render in renders.items():
        try:
            lengths[name] = len(render(value))
        except (TypeError, ValueError):
            pass
    return lengths


def test_output_size_bounds_every_rendering():
    named_zone = datetime.timezone(datetime.timedelta(hours=-23, minutes=-59, seconds=-59), "z'\"\\é" * 20)
    shared = ["é\U0001F600", b"\xff\x00", {1, "a"}]
    values = [
        None,
        False,
        -(10**200),
        -2.2250738585072014e-308,
        complex(-2.2250738585072014e-308, -1e308),
        ...,
        "",
        "q'\"\\" * 10,
        '"' * 3,
        "日本語 é \U0001F600 \x00 \u2028",
        b"\x00\xff" * 10,
        "é".encode() * 5,
        bytearray(b"\xff"),
        set(),
        frozenset(),
        frozenset({frozenset({1}), (1, "\\")}),
        {1: "a", None: 2.5, (1, 2): ["é"], "k'\"": True},
        datetime.date(9999, 12, 31),
        datetime.timedelta(days=-999999999, seconds=86399, microseconds=999999),
        datetime.datetime(9999, 12, 31, 23, 59, 59, 999999, fold=1),
        datetime.datetime(9999, 12, 31, 23, 59, 59, 999999, tzinfo=named_zone, fold=1),
        datetime.time(23, 59, 59, 999999, tzinfo=zoneinfo.ZoneInfo("America/Argentina/ComodRivadavia"), fold=1),
        named_zone,
        datetime.timezone.utc,
        pathlib.PurePosixPath("/a'b/é\\c"),
        [shared, shared, (shared,), {"k": shared}],
    ]
    for value in values:
        bound = _output_size(value, 10**12)
        for name, length in renderings(value).items():
            assert bound >= length, f"{name} of {value!r} is {length} characters, over the bound of {bound}"


def test_large_results_within_the_output_limit_pass():
    code = """
def run(inputs):
    return {"text": "a" * (1024 * 1024), "numbers": list(range(100000))}
"""
    output = PythonMonty(code=code).execute({})
    assert len(output["content"]["text"]) == 1024 * 1024
    assert len(output["content"]["numbers"]) == 100000


def test_class_instance_result_passes_with_its_attributes():
    code = """
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def run(inputs):
    return Point(1, 2)
"""
    output = PythonMonty(code=code, is_optimized_for_agents=True).execute({})
    assert "Point" in output["content"]
    assert "'x': 1" in output["content"]


def test_result_of_an_unsupported_type_is_rejected():
    node = PythonMonty(code="def run(inputs):\n    return ValueError('not a result')\n")
    with pytest.raises(ToolExecutionException, match="cannot pass on"):
        node.execute({})


def test_host_environment_is_not_reachable(monkeypatch):
    monkeypatch.setenv("DYNAMIQ_MONTY_TEST_SECRET", "s3cret")
    code = """
import os
def run(inputs):
    return os.getenv("DYNAMIQ_MONTY_TEST_SECRET")
"""
    node = PythonMonty(code=code)
    with pytest.raises(ToolExecutionException) as excinfo:
        node.execute({})
    assert "s3cret" not in str(excinfo.value)


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
