import pytest

from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.nodes.validators import ValidPython
from dynamiq.runnables import RunnableResult, RunnableStatus


@pytest.mark.parametrize(
    "content",
    [
        "def add(a, b):\n    return a + b",
        "for i in range(10):\n    print(i)",
        "class MyClass:\n    def __init__(self, value):\n        "
        "self.value = value\n\n    def get_value(self):\n        "
        "return self.value",
        "squares = [x**2 for x in range(10)]",
        "import math\nprint(math.sqrt(16))",
    ],
)
def test_workflow_with_valid_python(content):
    wf_valid_python = Workflow(
        flow=Flow(nodes=[ValidPython()]),
    )
    input_data = {"content": content}
    response = wf_valid_python.run(input_data=input_data)

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output={"valid": True, **input_data},
    ).to_dict()

    expected_output = {wf_valid_python.flow.nodes[0].id: expected_result}
    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output=expected_output,
    )
