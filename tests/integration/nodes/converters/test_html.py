from io import BytesIO
from pathlib import Path

import pytest

from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.nodes.converters.html import HTMLConverter
from dynamiq.nodes.node import NodeDependency
from dynamiq.nodes.utils import Output
from dynamiq.runnables import RunnableStatus


def write_html_to_path(path: Path, content: str) -> str:
    path.write_text(content)
    return str(path)


def write_html_to_bytesio(content: str, filename: str = "file.html") -> BytesIO:
    html_buffer = BytesIO(content.encode("utf-8"))
    html_buffer.name = filename
    return html_buffer


@pytest.fixture
def html_node():
    return HTMLConverter(
        id="html_converter",
        name="Test HTML Converter",
    )


@pytest.fixture
def output_node(html_node):
    return Output(id="output_node", depends=[NodeDependency(html_node)])


@pytest.fixture
def workflow(html_node, output_node):
    return Workflow(
        id="test_workflow",
        flow=Flow(
            nodes=[html_node, output_node],
        ),
        version="1",
    )


@pytest.fixture
def basic_html_content():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test HTML Document</title>
    </head>
    <body>
        <h1>Hello, World!</h1>
        <p>This is a test paragraph for the HTML converter.</p>
        <ul>
            <li>Item 1</li>
            <li>Item 2</li>
            <li>Item 3</li>
        </ul>
    </body>
    </html>
    """


@pytest.fixture
def basic_html_file_path(tmp_path, basic_html_content):
    test_file = tmp_path / "test.html"
    return write_html_to_path(test_file, basic_html_content)


@pytest.fixture
def basic_html_bytesio(basic_html_content):
    return write_html_to_bytesio(basic_html_content, "test.html")


@pytest.fixture
def complex_html_content():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Complex HTML Document</title>
    </head>
    <body>
        <h1>Complex HTML Test</h1>
        <p>This is a <strong>bold text</strong> and <em>italic text</em> in a paragraph.</p>

        <h2>Code Example</h2>
        <pre><code>def hello_world():
    print("Hello, World!")
        </code></pre>

        <h2>Table Example</h2>
        <table>
            <tr>
                <th>Header 1</th>
                <th>Header 2</th>
            </tr>
            <tr>
                <td>Cell 1</td>
                <td>Cell 2</td>
            </tr>
            <tr>
                <td>Cell 3</td>
                <td>Cell 4</td>
            </tr>
        </table>

        <h2>Link Example</h2>
        <p>Visit <a href="https://example.com">Example Website</a> for more information.</p>

        <blockquote>
            This is a blockquote with some text.
        </blockquote>
    </body>
    </html>
    """


@pytest.fixture
def complex_html_file_path(tmp_path, complex_html_content):
    test_file = tmp_path / "complex.html"
    return write_html_to_path(test_file, complex_html_content)


@pytest.fixture
def complex_html_bytesio(complex_html_content):
    return write_html_to_bytesio(complex_html_content, "complex.html")


@pytest.fixture
def invalid_html_content():
    return b"\x00\x01\x02\x03\x04This is not valid HTML content\xFF\xFE"


@pytest.fixture
def invalid_html_file_path(tmp_path, invalid_html_content):
    test_file = tmp_path / "test_file.html"
    test_file.write_bytes(invalid_html_content)
    return str(test_file)


@pytest.fixture
def invalid_html_bytesio(invalid_html_content):
    invalid_buffer = BytesIO(invalid_html_content)
    invalid_buffer.name = "invalid.html"
    return invalid_buffer


@pytest.fixture
def empty_html_file_path(tmp_path):
    empty_file = tmp_path / "empty_file.html"
    empty_file.touch()
    return str(empty_file)


@pytest.fixture
def empty_html_bytesio():
    empty_buffer = BytesIO()
    empty_buffer.name = "empty.html"
    return empty_buffer


@pytest.fixture
def non_existent_html_file(tmp_path):
    return str(tmp_path / "non_existent_file.html")


@pytest.mark.parametrize(
    "input_type,input_fixture",
    [
        ("file_paths", "basic_html_file_path"),
        ("files", "basic_html_bytesio"),
    ],
)
def test_workflow_with_html_converter(request, input_type, input_fixture):
    html_input = request.getfixturevalue(input_fixture)
    wf_html = Workflow(
        flow=Flow(nodes=[HTMLConverter()]),
    )

    input_data = {input_type: [html_input]}
    response = wf_html.run(input_data=input_data)

    assert response.status == RunnableStatus.SUCCESS

    node_id = wf_html.flow.nodes[0].id
    assert "documents" in response.output[node_id]["output"]
    assert len(response.output[node_id]["output"]["documents"]) == 1

    document = response.output[node_id]["output"]["documents"][0]
    assert "Hello, World!" in document["content"]
    assert "This is a test paragraph for the HTML converter" in document["content"]
    assert "Item 1" in document["content"]
    assert "Item 2" in document["content"]
    assert "Item 3" in document["content"]

    expected_source = html_input if input_type == "file_paths" else html_input.name
    assert document["metadata"]["file_path"] == expected_source


@pytest.mark.parametrize(
    "input_type,input_fixture",
    [
        ("file_paths", "complex_html_file_path"),
        ("files", "complex_html_bytesio"),
    ],
)
def test_html_converter_with_complex_content(request, input_type, input_fixture):
    html_input = request.getfixturevalue(input_fixture)
    wf_html = Workflow(
        flow=Flow(nodes=[HTMLConverter()]),
    )

    input_data = {input_type: [html_input]}
    response = wf_html.run(input_data=input_data)

    assert response.status == RunnableStatus.SUCCESS

    node_id = wf_html.flow.nodes[0].id
    assert "documents" in response.output[node_id]["output"]
    assert len(response.output[node_id]["output"]["documents"]) == 1

    document = response.output[node_id]["output"]["documents"][0]
    content = document["content"]
    assert "Complex HTML Test" in content
    assert "bold text" in content
    assert "italic text" in content
    assert "Code Example" in content
    assert "def hello_world()" in content
    assert "Table Example" in content
    assert "Header 1" in content
    assert "Header 2" in content
    assert "Example Website" in content
    assert "https://example.com" in content
    assert "blockquote" in content

    expected_source = html_input if input_type == "file_paths" else html_input.name
    assert document["metadata"]["file_path"] == expected_source


@pytest.mark.parametrize(
    "input_type,input_fixture",
    [
        ("file_paths", "invalid_html_file_path"),
        ("files", "invalid_html_bytesio"),
    ],
)
def test_workflow_with_html_node_failure(request, workflow, html_node, output_node, input_type, input_fixture):
    html_input = request.getfixturevalue(input_fixture)
    input_data = {input_type: [html_input]}

    result = workflow.run(input_data=input_data)

    assert result.status == RunnableStatus.SUCCESS

    html_result = result.output[html_node.id]
    assert html_result["status"] == RunnableStatus.FAILURE.value
    assert "error" in html_result

    output_result = result.output[output_node.id]
    assert output_result["status"] == RunnableStatus.SKIP.value


def test_workflow_with_html_node_file_not_found(workflow, html_node, output_node, non_existent_html_file):
    input_data = {"file_paths": [non_existent_html_file]}

    result = workflow.run(input_data=input_data)

    assert result.status == RunnableStatus.SUCCESS

    html_result = result.output[html_node.id]
    assert html_result["status"] == RunnableStatus.FAILURE.value
    assert "No files found in the provided paths" in html_result["error"]["message"]

    output_result = result.output[output_node.id]
    assert output_result["status"] == RunnableStatus.SKIP.value


@pytest.mark.parametrize(
    "input_type,input_fixture",
    [
        ("file_paths", "empty_html_file_path"),
        ("files", "empty_html_bytesio"),
    ],
)
def test_workflow_with_html_node_empty_file(request, workflow, html_node, output_node, input_type, input_fixture):
    html_input = request.getfixturevalue(input_fixture)
    input_data = {input_type: [html_input]}

    result = workflow.run(input_data=input_data)

    assert result.status == RunnableStatus.SUCCESS

    html_result = result.output[html_node.id]
    assert html_result["status"] == RunnableStatus.FAILURE.value
    assert "error" in html_result
    assert "Document is empty" in html_result["error"]["message"]

    output_result = result.output[output_node.id]
    assert output_result["status"] == RunnableStatus.SKIP.value
