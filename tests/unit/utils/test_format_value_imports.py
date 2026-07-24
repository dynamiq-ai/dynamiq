"""Guards on format_value: no heavy imports, scalar fast path, exception shape.

The import guard is the important one. `format_value` used to do a lazy
`from dynamiq.nodes.tools.python import PythonInputSchema` purely to satisfy an
isinstance() check. That import pulls in the entire tools/provider dependency tree
(litellm, openai, anthropic, playwright, stagehand, daytona, e2b, neo4j, boto3,
tiktoken -- ~4.3k modules) and it fired on the first traced value of *any* run, via
Flow._get_output -> RunnableResult.to_dict -> format_value. Measured cost was ~5.4s
on the first flow execution in a process; every cold serverless invocation paid it.
"""
import subprocess
import sys
import textwrap

import pytest

from dynamiq.utils import format_value

# Provider/tool SDKs that must not be dragged in by a bare flow execution.
HEAVY_MODULES = [
    "litellm",
    "openai",
    "anthropic",
    "playwright",
    "stagehand",
    "daytona",
    "e2b",
    "neo4j",
    "tiktoken",
]


def test_running_a_flow_does_not_import_provider_sdks():
    """A trivial flow run must not import the tool/provider dependency tree.

    Runs in a subprocess because the check is about *module import state*, which the
    rest of the test session has already polluted.
    """
    script = textwrap.dedent(
        f"""
        import sys, json
        from dynamiq import Workflow
        from dynamiq.flows import Flow
        from dynamiq.nodes.node import Node
        from dynamiq.nodes.types import NodeGroup

        class T(Node):
            group: NodeGroup = NodeGroup.UTILS
            name: str = "T"
            def execute(self, input_data, config=None, **kwargs):
                return {{"content": "ok"}}

        Workflow(flow=Flow(nodes=[T(id="a")])).run(input_data={{}})
        print(json.dumps([m for m in {HEAVY_MODULES!r} if m in sys.modules]))
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=180
    )
    assert proc.returncode == 0, proc.stderr
    leaked = __import__("json").loads(proc.stdout.strip().splitlines()[-1])
    assert leaked == [], (
        f"Running a trivial flow imported heavy provider SDKs: {leaked}. "
        "Something on the run path is importing dynamiq.nodes.tools/agents eagerly; "
        "this costs seconds on every cold start."
    )


@pytest.mark.parametrize("value", ["text", 42, 3.14, True, False, None])
def test_scalars_pass_through_unchanged(value):
    assert format_value(value) is value


def test_scalars_still_formatted_when_forced():
    """force_format_types must defeat the fast path."""
    assert format_value("text", force_format_types={str}) == "text"


def test_enum_serialization_is_unchanged():
    """Enums must serialize exactly as before the scalar fast path was added.

    The fast path keys on `type(value) in {str, int, ...}` -- an exact-type check, not
    isinstance -- specifically so str/int-subclassed enums keep falling through to the
    RootModel branch below it. That branch returns the enum member itself; this test
    pins that behaviour so the fast path can never start swallowing enums.
    """
    import enum

    class Color(str, enum.Enum):
        RED = "red"

    class Num(enum.IntEnum):
        ONE = 1

    assert format_value(Color.RED) is Color.RED
    assert format_value(Num.ONE) is Num.ONE


def test_exception_returns_a_dict_not_a_tuple():
    """Regression: this branch used to return `(dict, {})` while every other returns a bare value."""
    formatted = format_value(ValueError("boom"))
    assert isinstance(formatted, dict), f"expected dict, got {type(formatted).__name__}"
    assert formatted == {"message": "boom", "type": "ValueError", "recoverable": False}


def test_runnable_result_still_uses_its_own_to_dict():
    """The passthrough marker must preserve the old isinstance-based behaviour."""
    from dynamiq.runnables import RunnableResult, RunnableStatus

    result = RunnableResult(status=RunnableStatus.SUCCESS, input={"a": 1}, output={"b": 2})
    assert format_value(result) == {"status": "success", "input": {"a": 1}, "output": {"b": 2}}


def test_python_input_schema_still_uses_its_own_to_dict():
    from dynamiq.nodes.tools.python import PythonInputSchema

    assert format_value(PythonInputSchema(x=1, y="z")) == {"x": 1, "y": "z"}
