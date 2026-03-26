from io import BytesIO

import pytest
from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.runnables import RunnableResult, RunnableStatus
from dynamiq.types import Document

from dynamiq.nodes.extractors import ByIndexExtractor, ByRegexExtractor, FileTypeExtractor


@pytest.mark.parametrize(
    "input_list, index, expected_result",
    [
        ([1, 2, 3, 4, 5], 0, 1),
        ([5, 4, 3, 2, 1], 1, 4),
        ([1.2, 3.4, 5.6, 7.8], 2, 5.6),
        ([7.8, 5.6, 3.4, 1.2], 3, 1.2),
        (["apple", "banana", "cherry", "date"], 0, "apple"),
        (["zebra", "yak", "apple"], 2, "apple"),
        (
            [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}, {"name": "Charlie", "age": 35}],
            1,
            {"name": "Bob", "age": 25},
        ),
        (
            [{"name": "Charlie", "age": 35}, {"name": "Alice", "age": 30}],
            0,
            {"name": "Charlie", "age": 35},
        ),
        ([5, "apple", 3.2], None, "apple"),
        (["banana", 2, 4.5], 0, "banana"),
        (
            [
                Document(id="1", content="Document 1", embedding=[0.1, 0.1, 0.2], score=0.4),
                Document(id="2", content="Document 2", embedding=[0.1, 0.1, 0.2], score=0.8),
            ],
            1,
            Document(id="2", content="Document 2", embedding=[0.1, 0.1, 0.2], score=0.8),
        ),
    ],
)
def test_workflow_with_take_get_by_index(input_list, index, expected_result):
    wf_get_by_index = Workflow(flow=Flow(nodes=[ByIndexExtractor(index=1)]))

    input_data = {"input": input_list, "index": index}
    output = {"output": expected_result}
    response = wf_get_by_index.run(input_data=input_data)

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output=output,
    ).to_dict(skip_format_types={bytes})

    expected_output = {wf_get_by_index.flow.nodes[0].id: expected_result}
    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output=expected_output,
    )


def create_bytesio_with_name(content, name):
    file = BytesIO(content)
    file.name = name
    return file


@pytest.mark.parametrize(
    "filename, file, result",
    [
        ("image.png", None, "image"),
        ("document.docx", None, "document"),
        ("presentation.pptx", None, "presentation"),
        ("file.pdf", None, "pdf"),
        ("spreadsheet.xlsx", None, "spreadsheet"),
        ("archive.zip", None, "archive"),
        ("audio.mp3", None, "audio"),
        ("video.mp4", None, "video"),
        ("image.gif", None, "image"),
        ("unknownfile.xyz", None, None),
        (None, create_bytesio_with_name(b"file content with name", "unknownfile.xyz"), None),
        (None, create_bytesio_with_name(b"image content", "file_with_name.jpg"), "image"),
        (None, create_bytesio_with_name(b"spreadsheet content", "test.xlsx"), "spreadsheet"),
        (None, create_bytesio_with_name(b"presentation content", "test.pptx"), "presentation"),
        ("image.heic", None, "image"),
        ("archive.dcm", None, "archive"),
        ("archive.rpm", None, "archive"),
        ("audio.aiff", None, "audio"),
        ("video.mpg", None, "video"),
        ("font.otf", None, "font"),
        ("ebook.epub", None, "ebook"),
        (None, create_bytesio_with_name(b"audio content", "test.aiff"), "audio"),
        (None, create_bytesio_with_name(b"video content", "test.mpg"), "video"),
        (None, create_bytesio_with_name(b"font content", "test.otf"), "font"),
        (None, create_bytesio_with_name(b"executable content", "test.exe"), "executable"),
        (None, create_bytesio_with_name(b"ebook content", "test.epub"), "ebook"),
        (None, create_bytesio_with_name(b"unknown content", "unknownfile.xyz"), None),
    ],
)
def test_workflow_with_file_type_extraction(filename, file, result):
    wf_file_type_extraction = Workflow(flow=Flow(nodes=[FileTypeExtractor()]))

    input_data = {"filename": filename, "file": file}
    output = {"type": result}
    response = wf_file_type_extraction.run(input_data=input_data)

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output=output,
    ).to_dict(skip_format_types={BytesIO})

    expected_output = {wf_file_type_extraction.flow.nodes[0].id: expected_result}
    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output=expected_output,
    )


@pytest.mark.parametrize(
    "value, pattern, result",
    [
        ("My phone is 123-456-7890", r"\d{3}-\d{3}-\d{4}", ["123-456-7890"]),
        (
            "Emails: test@example.com, hello@domain.net",
            r"[\w\.-]+@[\w\.-]+\.\w+",
            ["test@example.com", "hello@domain.net"],
        ),
        ("No match here!", r"\d+", []),
        ("", r"\w+", []),
        ("####", r"#\w+", []),
        ("a1b2c3", r"[a-z]\d", ["a1", "b2", "c3"]),
    ],
)
def test_workflow_with_regex_extraction(value, pattern, result):
    wf_regex_extraction = Workflow(flow=Flow(nodes=[ByRegexExtractor()]))

    input_data = {"value": value, "pattern": pattern}
    output = {"matches": result}
    response = wf_regex_extraction.run(input_data=input_data)

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output=output,
    ).to_dict()

    expected_output = {wf_regex_extraction.flow.nodes[0].id: expected_result}
    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output=expected_output,
    )
