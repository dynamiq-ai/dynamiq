import pytest
from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.runnables import RunnableResult, RunnableStatus
from dynamiq.types import Document

from dynamiq.nodes.transformers import AnyToJSON, JSONToAny


@pytest.mark.parametrize(
    "value, result",
    [
        ({"name": "John Doe", "age": 30, "city": "New York"}, '{"name": "John Doe", "age": 30, "city": "New York"}'),
        (
            {"name": "Alice", "age": 25, "city": "Los Angeles", "isEmployed": True},
            '{"name": "Alice", "age": 25, "city": "Los Angeles", "isEmployed": true}',
        ),
        (42, "42"),
        (3.14, "3.14"),
        (True, "true"),
        (False, "false"),
        (None, "null"),
        ("Hello, world!", '"Hello, world!"'),
        ([1, 2, 3], "[1, 2, 3]"),
        ((4, 5, 6), "[4, 5, 6]"),
        (["apple", "banana", "cherry"], '["apple", "banana", "cherry"]'),
        ({}, "{}"),
        ([], "[]"),
        ("", '""'),
        (
            Document(id="1", content="Sample text", metadata={}, embedding=[], score=0.0),
            '{"id":"1","content":"Sample text","metadata":{},"embedding":[],"score":0.0}',
        ),
    ],
)
def test_workflow_with_any_to_json_string(value, result):
    wf_json_to_string = Workflow(flow=Flow(nodes=[AnyToJSON()]))

    input_data = {"value": value}
    output = {"content": result}
    response = wf_json_to_string.run(input_data=input_data)

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output=output,
    ).to_dict(skip_format_types={bytes})

    expected_output = {wf_json_to_string.flow.nodes[0].id: expected_result}
    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output=expected_output,
    )


@pytest.mark.parametrize(
    "value, result",
    [
        ('{"name": "Jane", "age": 22, "city": "Chicago"}', {"name": "Jane", "age": 22, "city": "Chicago"}),
        ('{"name": "Bob", "age": 28, "isEmployed": false}', {"name": "Bob", "age": 28, "isEmployed": False}),
    ],
)
def test_workflow_with_string_to_json(value, result):
    wf_json_to_string = Workflow(flow=Flow(nodes=[JSONToAny()]))

    input_data = {"value": value}
    output = {"content": result}
    response = wf_json_to_string.run(input_data=input_data)

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output=output,
    ).to_dict(skip_format_types={bytes})

    expected_output = {wf_json_to_string.flow.nodes[0].id: expected_result}
    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output=expected_output,
    )
