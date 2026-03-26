from io import BytesIO

import pytest
from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.nodes.types import ChoiceCondition, ConditionOperator
from dynamiq.runnables import RunnableResult, RunnableStatus

from dynamiq.nodes.filters import FileFilter, ListFilter


@pytest.mark.parametrize(
    "input_list, filters, expected_result",
    [
        (
            [1, 2, 3, 4, 5],
            ChoiceCondition(
                operator=ConditionOperator.NUMERIC_GREATER_THAN,
                variable="$",
                value=3,
                operands=[],
            ),
            [4, 5],
        ),
        (
            [1, 2, 3, 4, 5],
            ChoiceCondition(
                operator=ConditionOperator.NUMERIC_LESS_THAN_OR_EQUALS,
                variable="$",
                value=3,
                operands=[],
            ),
            [1, 2, 3],
        ),
        (
            [1.2, 3.4, 5.6, 7.8],
            ChoiceCondition(
                operator=ConditionOperator.NUMERIC_EQUALS,
                variable="$",
                value=5.6,
                operands=[],
            ),
            [5.6],
        ),
        (
            [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}, {"name": "Charlie", "age": 35}],
            ChoiceCondition(
                operator=ConditionOperator.AND,
                operands=[
                    ChoiceCondition(
                        operator=ConditionOperator.NUMERIC_GREATER_THAN,
                        variable="$.age",
                        value=25,
                        operands=[],
                    ),
                    ChoiceCondition(
                        operator=ConditionOperator.NUMERIC_LESS_THAN,
                        variable="$.age",
                        value=35,
                        operands=[],
                    ),
                ],
            ),
            [{"name": "Alice", "age": 30}],
        ),
        (
            [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}, {"name": "Charlie", "age": 35}],
            ChoiceCondition(
                operator=ConditionOperator.OR,
                operands=[
                    ChoiceCondition(
                        operator=ConditionOperator.NUMERIC_EQUALS,
                        variable="$.age",
                        value=25,
                        operands=[],
                    ),
                    ChoiceCondition(
                        operator=ConditionOperator.STRING_EQUALS,
                        variable="$.name",
                        value="Charlie",
                        operands=[],
                    ),
                ],
            ),
            [{"name": "Bob", "age": 25}, {"name": "Charlie", "age": 35}],
        ),
        (
            [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}, {"name": "Charlie", "age": 35}],
            ChoiceCondition(
                operator=ConditionOperator.NUMERIC_GREATER_THAN_OR_EQUALS,
                variable="$.age",
                value=30,
                operands=[],
            ),
            [{"name": "Alice", "age": 30}, {"name": "Charlie", "age": 35}],
        ),
        (
            [1, 2, 3],
            ChoiceCondition(
                operator=ConditionOperator.NUMERIC_GREATER_THAN,
                variable="$",
                value=4,
                operands=[],
            ),
            [],
        ),
    ],
)
def test_workflow_with_list_filter(input_list, filters, expected_result):
    wf_json_list_filter = Workflow(flow=Flow(nodes=[ListFilter(filters=filters)]))

    input_data = {"input": input_list}
    output = {"output": expected_result}
    response = wf_json_list_filter.run(input_data=input_data)

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output=output,
    ).to_dict(skip_format_types={bytes})

    expected_output = {wf_json_list_filter.flow.nodes[0].id: expected_result}
    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output=expected_output,
    )


@pytest.fixture
def create_test_files():
    """Creates a list of mock file objects with fixed attributes."""
    files = []

    test_data = [
        ("csv_example.csv", 1500, "csv"),
        ("txt_example.txt", 800, "txt"),
        ("pdf_example.pdf", 250000, "pdf"),
        ("image.png", 500000, "png"),
    ]

    for name, size, extension in test_data:
        file_obj = BytesIO()
        file_obj.name = name
        file_obj.size = size
        file_obj.content_type = extension
        files.append(file_obj)

    return files


@pytest.mark.parametrize(
    "filters, expected_files",
    [
        (
            ChoiceCondition(
                operator=ConditionOperator.AND,
                operands=[
                    ChoiceCondition(
                        operator=ConditionOperator.NUMERIC_GREATER_THAN,
                        variable="$.size",
                        value=1000,
                        operands=[],
                    )
                ],
            ),
            [
                {"size": 1500, "content_type": "csv", "filename": "csv_example.csv"},
                {"size": 250000, "content_type": "pdf", "filename": "pdf_example.pdf"},
                {"size": 500000, "content_type": "png", "filename": "image.png"},
            ],
        ),
        (
            ChoiceCondition(
                operator=ConditionOperator.OR,
                operands=[
                    ChoiceCondition(
                        operator=ConditionOperator.STRING_EQUALS,
                        variable="$.content_type",
                        value="txt",
                        operands=[],
                    ),
                    ChoiceCondition(
                        operator=ConditionOperator.STRING_EQUALS,
                        variable="$.content_type",
                        value="csv",
                        operands=[],
                    ),
                ],
            ),
            [
                {"size": 1500, "content_type": "csv", "filename": "csv_example.csv"},
                {"size": 800, "content_type": "txt", "filename": "txt_example.txt"},
            ],
        ),
        (
            ChoiceCondition(
                operator=ConditionOperator.STRING_EQUALS,
                variable="$.content_type",
                value="pdf",
                operands=[],
            ),
            [{"size": 250000, "content_type": "pdf", "filename": "pdf_example.pdf"}],
        ),
        (
            ChoiceCondition(
                operator=ConditionOperator.STRING_EQUALS,
                variable="$.content_type",
                value="png",
                operands=[],
            ),
            [
                {"size": 500000, "content_type": "png", "filename": "image.png"},
            ],
        ),
        (
            ChoiceCondition(
                operator=ConditionOperator.OR,
                operands=[
                    ChoiceCondition(
                        operator=ConditionOperator.STRING_EQUALS,
                        variable="$.content_type",
                        value="pdf",
                        operands=[],
                    ),
                    ChoiceCondition(
                        operator=ConditionOperator.NUMERIC_GREATER_THAN,
                        variable="$.size",
                        value=200000,
                        operands=[],
                    ),
                ],
            ),
            [
                {"size": 250000, "content_type": "pdf", "filename": "pdf_example.pdf"},
                {"size": 500000, "content_type": "png", "filename": "image.png"},
            ],
        ),
    ],
)
def test_workflow_with_file_filter(create_test_files, filters, expected_files):
    wf_file_filter = Workflow(flow=Flow(nodes=[FileFilter(filters=filters)]))

    input_data = {"files": create_test_files}
    result = [
        file
        for file in input_data["files"]
        if any(
            {"size": file.size, "content_type": file.content_type, "filename": file.name} == expected
            for expected in expected_files
        )
    ]
    output = {"output": result}
    response = wf_file_filter.run(input_data=input_data)

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output=output,
    ).to_dict(skip_format_types={BytesIO})

    expected_output = {wf_file_filter.flow.nodes[0].id: expected_result}
    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output=expected_output,
    )
