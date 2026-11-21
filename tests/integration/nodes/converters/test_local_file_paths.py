"""Converters and parsers read files by local path only where the environment allows it."""

from io import BytesIO
from unittest.mock import MagicMock

import pytest

from dynamiq import Workflow
from dynamiq.components.converters.text import TextFileConverter as TextFileConverterComponent
from dynamiq.connections import AWSTextract
from dynamiq.connections import MistralOCR as MistralOCRConnection
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.connections import Unstructured
from dynamiq.flows import Flow
from dynamiq.nodes.converters import (
    CSVConverter,
    DOCXFileConverter,
    ExcelFileConverter,
    HTMLConverter,
    LLMImageConverter,
    LLMPDFConverter,
    MultiFileTypeConverter,
    PPTXFileConverter,
    PyPDFConverter,
    TextFileConverter,
    TextractFileConverter,
    UnstructuredFileConverter,
)
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.node import InputTransformer
from dynamiq.nodes.parsers import MistralOCR
from dynamiq.runnables import RunnableStatus
from dynamiq.utils.local_files import LOCAL_FILE_PATHS_ALLOWED_ENV_VAR, LocalFilePathsNotAllowedError

CONTENT = "Quarterly figures: 4f1c9e"


def llm() -> OpenAI:
    return OpenAI(connection=OpenAIConnection(api_key="test-api-key"), model="gpt-4o-mini")


CONVERTERS = [
    pytest.param(TextFileConverter, id="text"),
    pytest.param(HTMLConverter, id="html"),
    pytest.param(PyPDFConverter, id="pdf"),
    pytest.param(DOCXFileConverter, id="docx"),
    pytest.param(PPTXFileConverter, id="pptx"),
    pytest.param(ExcelFileConverter, id="excel"),
    pytest.param(CSVConverter, id="csv"),
    pytest.param(MultiFileTypeConverter, id="multi-file-type"),
    pytest.param(lambda: LLMImageConverter(llm=llm()), id="llm-image"),
    pytest.param(lambda: LLMPDFConverter(llm=llm()), id="llm-pdf"),
    pytest.param(
        lambda: UnstructuredFileConverter(connection=Unstructured(url="test_url", api_key="test-api-key")),
        id="unstructured",
    ),
    pytest.param(
        lambda: TextractFileConverter(
            connection=AWSTextract(access_key_id="test-key", secret_access_key="test-secret", region="us-east-1"),
            client=MagicMock(spec=["analyze_document", "detect_document_text"]),
        ),
        id="textract",
    ),
]


@pytest.fixture
def local_file_paths_off(monkeypatch):
    monkeypatch.setenv(LOCAL_FILE_PATHS_ALLOWED_ENV_VAR, "false")


@pytest.fixture
def notes_path(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text(CONTENT)
    return str(path)


def assert_refused(result, node):
    assert result.status == RunnableStatus.FAILURE
    node_result = result.output[node.id]
    assert node_result["status"] == RunnableStatus.FAILURE.value
    assert LOCAL_FILE_PATHS_ALLOWED_ENV_VAR in node_result["error"]["message"]
    assert CONTENT not in str(result.output)


@pytest.mark.parametrize("make_converter", CONVERTERS)
def test_converter_refuses_local_paths_in_its_input_when_they_are_turned_off(
    local_file_paths_off, notes_path, make_converter
):
    converter = make_converter()
    workflow = Workflow(flow=Flow(nodes=[converter]))

    result = workflow.run(input_data={"file_paths": [notes_path]})

    assert_refused(result, converter)


def test_converter_refuses_local_paths_an_input_transformer_selects_when_they_are_turned_off(
    local_file_paths_off, notes_path
):
    converter = TextFileConverter(input_transformer=InputTransformer(selector={"file_paths": "$.documents"}))
    workflow = Workflow(flow=Flow(nodes=[converter]))

    result = workflow.run(input_data={"documents": [notes_path]})

    assert_refused(result, converter)


def test_converter_component_refuses_a_path_before_looking_for_it(local_file_paths_off, tmp_path):
    with pytest.raises(LocalFilePathsNotAllowedError):
        TextFileConverterComponent().run(file_paths=[str(tmp_path / "missing.txt")])


@pytest.mark.parametrize("make_converter", [TextFileConverter, CSVConverter])
def test_converter_reads_file_objects_when_local_paths_are_turned_off(local_file_paths_off, make_converter):
    converter = make_converter()
    file = BytesIO(f"note\n{CONTENT}\n".encode())
    file.name = "notes.csv"
    workflow = Workflow(flow=Flow(nodes=[converter]))

    result = workflow.run(input_data={"files": [file]})

    assert result.status == RunnableStatus.SUCCESS
    documents = result.output[converter.id]["output"]["documents"]
    assert CONTENT in " ".join(document["content"] for document in documents)


@pytest.mark.parametrize("make_converter", [TextFileConverter, CSVConverter])
def test_converter_reads_local_paths_when_the_environment_does_not_turn_them_off(monkeypatch, tmp_path, make_converter):
    monkeypatch.delenv(LOCAL_FILE_PATHS_ALLOWED_ENV_VAR, raising=False)
    path = tmp_path / "notes.csv"
    path.write_text(f"note\n{CONTENT}\n")
    converter = make_converter()
    workflow = Workflow(flow=Flow(nodes=[converter]))

    result = workflow.run(input_data={"file_paths": [str(path)]})

    assert result.status == RunnableStatus.SUCCESS
    documents = result.output[converter.id]["output"]["documents"]
    assert CONTENT in " ".join(document["content"] for document in documents)


def test_mistral_ocr_refuses_a_local_path_when_local_paths_are_turned_off(local_file_paths_off, notes_path):
    ocr = MistralOCR(connection=MistralOCRConnection(api_key="test-api-key"))
    workflow = Workflow(flow=Flow(nodes=[ocr]))

    result = workflow.run(input_data={"file": notes_path})

    assert_refused(result, ocr)
