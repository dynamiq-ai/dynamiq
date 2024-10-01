import pytest

from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.nodes.validators import ValidJSON
from dynamiq.runnables import RunnableResult, RunnableStatus


@pytest.mark.parametrize(
    "content",
    [
        {"name": "John", "age": 30, "city": "New York"},
        '{"person": {"name": "John", "age": 30, "address": {"city": "New York", "zipcode": "10021"}}}',
        '[{"name": "John", "age": 30}, {"name": "Jane", "age": 25}]',
        '{"string": "Hello", "number": 123, "boolean": true, "null": null, "array": [1, 2, 3], '
        '"object": {"key": "value"}}',
        "{}",
    ],
)
def test_workflow_with_valid_json(content):
    wf_valid_json = Workflow(
        flow=Flow(nodes=[ValidJSON()]),
    )
    input_data = {"content": content}
    response = wf_valid_json.run(input_data=input_data)

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output={"valid": True, **input_data},
    ).to_dict()

    expected_output = {wf_valid_json.flow.nodes[0].id: expected_result}
    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output=expected_output,
    )
