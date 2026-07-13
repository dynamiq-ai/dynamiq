import zipfile
from io import BytesIO

import pytest
from openpyxl import Workbook as ExcelWorkbook

from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.nodes.converters.csv import CSVConverter
from dynamiq.nodes.converters.excel import ExcelFileConverter
from dynamiq.nodes.converters.multi_file_type_converter import MultiFileTypeConverter
from dynamiq.runnables import RunnableStatus
from dynamiq.types import DocumentCreationMode


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


def build_empty_xlsx_bytesio(filename: str = "empty.xlsx") -> BytesIO:
    workbook = ExcelWorkbook()
    buffer = BytesIO()
    workbook.save(buffer)
    buffer.seek(0)
    buffer.name = filename
    return buffer


def build_csv_bytesio(filename: str = "test.csv") -> BytesIO:
    buffer = BytesIO(b"name,age\nAlice,30\nBob,25\n")
    buffer.name = filename
    return buffer


def build_ods_bytesio(filename: str = "extensionless-ods") -> BytesIO:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("mimetype", "application/vnd.oasis.opendocument.spreadsheet")
        archive.writestr("content.xml", "<office:document-content/>")
    buffer.seek(0)
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


@pytest.mark.parametrize(
    "mode",
    [
        DocumentCreationMode.ONE_DOC_PER_FILE,
        DocumentCreationMode.ONE_DOC_PER_ROW,
        DocumentCreationMode.ONE_DOC_PER_SHEET,
    ],
)
def test_excel_converter_rejects_empty_workbook_in_all_creation_modes(mode):
    converter = ExcelFileConverter(workbook_document_creation_mode=mode)
    workflow = Workflow(flow=Flow(nodes=[converter]))

    response = workflow.run(input_data={"files": [build_empty_xlsx_bytesio()]})

    assert response.status == RunnableStatus.FAILURE
    assert "contains no extractable content" in response.output[converter.id]["error"]["message"]


def test_excel_converter_creates_self_describing_xlsx_row_documents():
    converter = ExcelFileConverter(workbook_document_creation_mode="one-doc-per-row")
    documents = run_and_get_documents(converter, [build_xlsx_bytesio()])

    assert len(documents) == 2
    assert documents[0]["content"] == "Sheet: People\nname: Alice\nage: 30"
    assert documents[0]["metadata"]["sheet_name"] == "People"
    assert documents[0]["metadata"]["row_number"] == 2
    assert documents[0]["metadata"]["document_type"] == "table_row"
    assert documents[0]["metadata"]["source"] == "test.xlsx"


def test_excel_converter_can_create_one_document_per_sheet():
    converter = ExcelFileConverter(workbook_document_creation_mode="one-doc-per-sheet")
    documents = run_and_get_documents(converter, [build_xlsx_bytesio()])

    assert len(documents) == 1
    assert "| name | age |" in documents[0]["content"]
    assert documents[0]["metadata"]["sheet_name"] == "People"
    assert documents[0]["metadata"]["row_count"] == 2


def test_excel_converter_with_csv():
    documents = run_and_get_documents(ExcelFileConverter(), [build_csv_bytesio()])

    assert len(documents) == 1
    content = documents[0]["content"]
    assert "| name | age |" in content
    assert "| Alice | 30 |" in content


def test_excel_converter_creates_self_describing_csv_row_documents():
    converter = ExcelFileConverter(delimited_document_creation_mode="one-doc-per-row")
    documents = run_and_get_documents(converter, [build_csv_bytesio()])

    assert len(documents) == 2
    assert documents[0]["content"] == "name: Alice\nage: 30"
    assert documents[1]["content"] == "name: Bob\nage: 25"
    assert documents[0]["metadata"] == {
        "content_type": "text/csv",
        "document_type": "table_row",
        "file_path": "test.csv",
        "row_number": 2,
        "source": "test.csv",
    }


def test_csv_row_documents_preserve_original_source_url():
    converter = ExcelFileConverter(delimited_document_creation_mode="one-doc-per-row")
    converter.init_components()
    result = converter.run(
        input_data={
            "files": [build_csv_bytesio()],
            "metadata": {"dynamiq_item_source_provider_url": "https://example.com/pricing"},
        }
    )

    assert result.status == RunnableStatus.SUCCESS
    documents = result.output["documents"]
    assert all(
        document.metadata["dynamiq_item_source_provider_url"] == "https://example.com/pricing" for document in documents
    )
    assert all(document.metadata["source"] == "https://example.com/pricing" for document in documents)


def test_csv_row_documents_normalize_blank_and_duplicate_headers():
    buffer = BytesIO(b",plan,plan\nFeature,Free,Pro\n")
    buffer.name = "pricing.csv"
    documents = run_and_get_documents(
        ExcelFileConverter(delimited_document_creation_mode="one-doc-per-row"),
        [buffer],
    )

    assert documents[0]["content"] == "column_1: Feature\nplan: Free\nplan_2: Pro"


def test_spreadsheet_row_documents_keep_generated_headers_globally_unique():
    buffer = BytesIO(b"plan,plan,plan_2\nFree,Pro,Business\n")
    buffer.name = "plans.csv"

    documents = run_and_get_documents(
        ExcelFileConverter(delimited_document_creation_mode="one-doc-per-row"),
        [buffer],
    )

    assert documents[0]["content"] == "plan: Free\nplan_3: Pro\nplan_2: Business"


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


def test_multi_file_converter_can_route_csv_as_rows():
    documents = run_and_get_documents(
        MultiFileTypeConverter(converters=[CSVConverter()]),
        [build_csv_bytesio("pricing.csv")],
    )

    assert len(documents) == 2
    assert documents[0]["content"] == "name: Alice\nage: 30"


def test_multi_file_converter_can_route_xlsx_as_rows():
    documents = run_and_get_documents(
        MultiFileTypeConverter(converters=[ExcelFileConverter(workbook_document_creation_mode="one-doc-per-row")]),
        [build_xlsx_bytesio("pricing.xlsx")],
    )

    assert len(documents) == 2
    assert documents[0]["content"] == "Sheet: People\nname: Alice\nage: 30"
    assert documents[0]["metadata"]["file_type"] == "text"
    assert documents[0]["metadata"]["source_file_type"] == "spreadsheet"


def test_multi_file_converter_routes_extensionless_delimited_text_to_csv_converter():
    documents = run_and_get_documents(
        MultiFileTypeConverter(converters=[CSVConverter()]),
        [build_csv_bytesio("extensionless-item-id")],
    )

    assert len(documents) == 2
    assert documents[0]["content"] == "name: Alice\nage: 30"


def test_multi_file_converter_does_not_route_legacy_spreadsheet_to_excel():
    buffer = BytesIO(b"name,age\nAlice,30\n")
    buffer.name = "legacy.xls"

    workflow = Workflow(flow=Flow(nodes=[MultiFileTypeConverter()]))
    response = workflow.run(input_data={"files": [buffer]})

    assert response.status == RunnableStatus.FAILURE
    node_id = workflow.flow.nodes[0].id
    assert "Unsupported file type: None" in response.output[node_id]["error"]["message"]
    assert "ExcelFileConverter" not in response.output[node_id]["error"]["message"]


def test_multi_file_converter_does_not_route_extensionless_ods_to_excel():
    workflow = Workflow(flow=Flow(nodes=[MultiFileTypeConverter()]))
    response = workflow.run(input_data={"files": [build_ods_bytesio()]})

    assert response.status == RunnableStatus.FAILURE
    node_id = workflow.flow.nodes[0].id
    assert "Unsupported file type: None" in response.output[node_id]["error"]["message"]
    assert "ExcelFileConverter" not in response.output[node_id]["error"]["message"]


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
    assert documents[0]["metadata"]["source_file_type"] == "spreadsheet"
    assert documents[0]["metadata"]["file_type"] == "markdown"
    assert documents[1]["metadata"]["source_file_type"] == "text"
    assert documents[1]["metadata"]["file_type"] == "text"
