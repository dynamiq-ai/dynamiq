import pytest

from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.nodes.validators import ValidChoices
from dynamiq.runnables import RunnableResult, RunnableStatus


@pytest.mark.parametrize(
    ("content", "choices"),
    [
        (3, [3, 4, 5, 7, 9]),
        ("ch", ["ch1", "ch", "ch2"]),
        (True, [True, False]),
        (9.5, [8.5, 9.5, 10.5]),
        ("apple", ["apple", "banana", "cherry"]),
    ],
)
def test_workflow_with_valid_choices(content, choices):
    wf_valid_choices = Workflow(
        flow=Flow(nodes=[ValidChoices(choices=choices)]),
    )
    input_data = {"content": content}
    response = wf_valid_choices.run(input_data=input_data)

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output={"valid": True, **input_data},
    ).to_dict()

    expected_output = {wf_valid_choices.flow.nodes[0].id: expected_result}
    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output=expected_output,
    )
