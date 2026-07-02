from io import BytesIO

import pytest
from openpyxl import Workbook as ExcelWorkbook

from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.nodes.converters.excel import ExcelFileConverter
from dynamiq.nodes.converters.multi_file_type_converter import MultiFileTypeConverter
from dynamiq.runnables import RunnableStatus


def build_xlsx_bytesio(filename: str = "test.xlsx") -> BytesIO:
    workbook = ExcelWorkbook()
    sheet = workbook.active
    sheet.title = "People"
    sheet.append(["name", "age"])
    sheet.append(["Alice", 30])
    sheet.append(["Bob", 25])
    buffer = BytesIO()
    workbook.save(buffer)
    buffer.seek(0)
    buffer.name = filename
    return buffer


def build_csv_bytesio(filename: str = "test.csv") -> BytesIO:
    buffer = BytesIO(b"name,age\nAlice,30\nBob,25\n")
    buffer.name = filename
    return buffer


def run_and_get_documents(node, files):
    workflow = Workflow(flow=Flow(nodes=[node]))
    response = workflow.run(input_data={"files": files})
    assert response.status == RunnableStatus.SUCCESS
    return response.output[node.id]["output"]["documents"]


def test_excel_converter_with_xlsx():
    documents = run_and_get_documents(ExcelFileConverter(), [build_xlsx_bytesio()])

    assert len(documents) == 1
    content = documents[0]["content"]
    assert "| name | age |" in content
    assert "| Alice | 30 |" in content
    assert "| Bob | 25 |" in content
    assert documents[0]["metadata"]["file_path"] == "test.xlsx"


def test_excel_converter_with_csv():
    documents = run_and_get_documents(ExcelFileConverter(), [build_csv_bytesio()])

    assert len(documents) == 1
    content = documents[0]["content"]
    assert "| name | age |" in content
    assert "| Alice | 30 |" in content


def test_excel_converter_with_extensionless_csv():
    documents = run_and_get_documents(ExcelFileConverter(), [build_csv_bytesio("extensionless-item-id")])

    assert len(documents) == 1
    content = documents[0]["content"]
    assert "| name | age |" in content
    assert "| Alice | 30 |" in content


@pytest.mark.parametrize("filename", ["legacy.xls", "open_document.ods"])
def test_excel_converter_rejects_unsupported_spreadsheet_extensions(filename):
    buffer = BytesIO(b"legacy spreadsheet bytes")
    buffer.name = filename

    workflow = Workflow(flow=Flow(nodes=[ExcelFileConverter()]))
    response = workflow.run(input_data={"files": [buffer]})

    assert response.status == RunnableStatus.FAILURE
    node_id = workflow.flow.nodes[0].id
    assert "Unsupported spreadsheet extension" in response.output[node_id]["error"]["message"]


def test_multi_file_converter_routes_spreadsheets_locally():
    documents = run_and_get_documents(
        MultiFileTypeConverter(),
        [build_xlsx_bytesio("routed.xlsx"), build_csv_bytesio("routed.csv")],
    )

    assert len(documents) == 2
    for document in documents:
        assert "| Alice | 30 |" in document["content"]


def test_multi_file_converter_does_not_route_legacy_spreadsheet_to_excel():
    buffer = BytesIO(b"name,age\nAlice,30\n")
    buffer.name = "legacy.xls"

    workflow = Workflow(flow=Flow(nodes=[MultiFileTypeConverter()]))
    response = workflow.run(input_data={"files": [buffer]})

    assert response.status == RunnableStatus.FAILURE
    node_id = workflow.flow.nodes[0].id
    assert "Unsupported file type: None" in response.output[node_id]["error"]["message"]


def test_multi_file_converter_routes_extensionless_csv_to_excel():
    documents = run_and_get_documents(MultiFileTypeConverter(), [build_csv_bytesio("extensionless-item-id")])

    assert len(documents) == 1
    content = documents[0]["content"]
    assert "| name | age |" in content
    assert "| Alice | 30 |" in content


def test_multi_file_converter_detects_type_without_extension():
    # No extension in the name and no filename metadata: content sniffing routes xlsx/text.
    xlsx_file = build_xlsx_bytesio("extensionless-item-id")
    text_file = BytesIO(b"Plain text knowledge base item without extension.")
    text_file.name = "another-extensionless-item-id"

    documents = run_and_get_documents(MultiFileTypeConverter(), [xlsx_file, text_file])

    assert len(documents) == 2
    assert "| Alice | 30 |" in documents[0]["content"]
    assert "Plain text knowledge base item" in documents[1]["content"]
