from io import BytesIO

import pytest
from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.runnables import RunnableResult, RunnableStatus

from dynamiq.nodes.transformers import CSVToList, ExcelToList
from dynamiq.nodes.transformers.csv import CSVToListTransformerInputSchema
from dynamiq.nodes.transformers.excel import ExcelToListTransformerInputSchema


@pytest.mark.parametrize(
    "value,delimiter, result",
    [
        (
            BytesIO(b"name;age;city\nJohn Doe;30;New York\nJane Smith;25;Los Angeles\nAlice Johnson;35;Chicago\n"),
            ";",
            [
                {"name": "John Doe", "age": "30", "city": "New York"},
                {"name": "Jane Smith", "age": "25", "city": "Los Angeles"},
                {"name": "Alice Johnson", "age": "35", "city": "Chicago"},
            ],
        ),
        (
            BytesIO(b"name,age,city,isEmployed\nJohn,30,New York,True\nJane,25,Los Angeles,False\n"),
            None,
            [
                {"name": "John", "age": "30", "city": "New York", "isEmployed": "True"},
                {"name": "Jane", "age": "25", "city": "Los Angeles", "isEmployed": "False"},
            ],
        ),
        (
            "name,age,city,job\nAlice,28,San Francisco,Engineer\nBob,35,New York,Doctor\n",
            None,
            [
                {"name": "Alice", "age": "28", "city": "San Francisco", "job": "Engineer"},
                {"name": "Bob", "age": "35", "city": "New York", "job": "Doctor"},
            ],
        ),
        (
            "name;age;city\nJohn Doe;30;New York\nJane Smith;25;Los Angeles\nAlice Johnson;35;Chicago\n",
            ";",
            [
                {"name": "John Doe", "age": "30", "city": "New York"},
                {"name": "Jane Smith", "age": "25", "city": "Los Angeles"},
                {"name": "Alice Johnson", "age": "35", "city": "Chicago"},
            ],
        ),
    ],
)
def test_workflow_with_csv_to_json(value, delimiter, result):
    wf_csv_to_json = Workflow(flow=Flow(nodes=[CSVToList()]))

    delimiter = {"delimiter": delimiter} if delimiter else {}
    input_data = {"value": value, **delimiter}
    output = {"content": result}
    response = wf_csv_to_json.run(input_data=input_data)

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=dict(CSVToListTransformerInputSchema(**input_data)),
        output=output,
    ).to_dict(skip_format_types={BytesIO, bytes})

    expected_output = {wf_csv_to_json.flow.nodes[0].id: expected_result}
    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output=expected_output,
    )


@pytest.fixture
def create_excel_binary(sheet_name="Sheet_name"):
    import pandas as pd

    data = [
        {"name": "John Doe", "age": 30, "city": "New York", "isEmployed": True},
        {"name": "Jane Smith", "age": 25, "city": "Los Angeles", "isEmployed": True},
        {"name": "Alice Johnson", "age": 35, "city": "Chicago", "isEmployed": False},
    ]
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame(data).to_excel(writer, index=False, sheet_name=sheet_name)
    buffer.seek(0)
    return buffer


@pytest.mark.parametrize(
    "sheet_name, result",
    [
        (
            "Sheet_name",
            {
                "Sheet_name": [
                    {"name": "John Doe", "age": 30, "city": "New York", "isEmployed": True},
                    {"name": "Jane Smith", "age": 25, "city": "Los Angeles", "isEmployed": True},
                    {"name": "Alice Johnson", "age": 35, "city": "Chicago", "isEmployed": False},
                ]
            },
        ),
        (
            None,
            {
                "Sheet_name": [
                    {"name": "John Doe", "age": 30, "city": "New York", "isEmployed": True},
                    {"name": "Jane Smith", "age": 25, "city": "Los Angeles", "isEmployed": True},
                    {"name": "Alice Johnson", "age": 35, "city": "Chicago", "isEmployed": False},
                ]
            },
        ),
    ],
)
def test_workflow_with_excel_to_json(create_excel_binary, sheet_name, result):
    wf_json_to_csv = Workflow(flow=Flow(nodes=[ExcelToList()]))
    sheet_name = {"sheet_name": sheet_name} if sheet_name else {}
    input_data = {"value": create_excel_binary, **sheet_name}
    output = {"content": result}
    response = wf_json_to_csv.run(input_data=input_data)

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=dict(ExcelToListTransformerInputSchema(**input_data)),
        output=output,
    ).to_dict(skip_format_types={BytesIO, bytes})

    expected_output = {wf_json_to_csv.flow.nodes[0].id: expected_result}
    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output=expected_output,
    )
