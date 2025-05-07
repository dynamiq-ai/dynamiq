import os
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest

from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.nodes.converters.html import HTMLConverter
from dynamiq.nodes.node import NodeDependency
from dynamiq.nodes.utils import Output
from dynamiq.runnables import RunnableStatus


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


def test_workflow_with_html_converter():
    wf_html = Workflow(
        flow=Flow(nodes=[HTMLConverter()]),
    )

    html_content = """
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

    file = BytesIO(html_content.encode("utf-8"))
    file.name = "test.html"

    input_data = {"files": [file]}
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
    assert document["metadata"]["file_path"] == "test.html"


def test_html_converter_with_complex_content():
    wf_html = Workflow(
        flow=Flow(nodes=[HTMLConverter()]),
    )

    html_content = """
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

    file = BytesIO(html_content.encode("utf-8"))
    file.name = "complex.html"

    input_data = {"files": [file]}
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


def test_workflow_with_html_node_failure(workflow, html_node, output_node, tmp_path):
    test_file = tmp_path / "test_file.html"
    test_file.write_text("Not valid HTML content")

    with patch(
        "dynamiq.components.converters.html.lxml_html.fromstring", side_effect=Exception("Failed to parse HTML content")
    ):
        input_data = {"file_paths": [str(test_file)]}

        result = workflow.run(input_data=input_data)

        assert result.status == RunnableStatus.SUCCESS

        html_result = result.output[html_node.id]
        assert html_result["status"] == RunnableStatus.FAILURE.value
        assert "Failed to parse HTML content" in html_result["error"]["message"]

        output_result = result.output[output_node.id]
        assert output_result["status"] == RunnableStatus.SKIP.value


def test_workflow_with_html_node_file_not_found(workflow, html_node, output_node):
    non_existent_path = str(Path("/tmp") / f"non_existent_file_{os.getpid()}.html")
    input_data = {"file_paths": [non_existent_path]}

    result = workflow.run(input_data=input_data)

    assert result.status == RunnableStatus.SUCCESS

    html_result = result.output[html_node.id]
    assert html_result["status"] == RunnableStatus.FAILURE.value
    assert "No files found in the provided paths" in html_result["error"]["message"]

    output_result = result.output[output_node.id]
    assert output_result["status"] == RunnableStatus.SKIP.value


def test_workflow_with_html_node_empty_file(workflow, html_node, output_node, tmp_path):
    empty_file = tmp_path / "empty_file.html"
    empty_file.touch()

    input_data = {"file_paths": [str(empty_file)]}

    result = workflow.run(input_data=input_data)

    assert result.status == RunnableStatus.SUCCESS

    html_result = result.output[html_node.id]
    if html_result["status"] == RunnableStatus.FAILURE.value:
        output_result = result.output[output_node.id]
        assert output_result["status"] == RunnableStatus.SKIP.value
    else:
        assert "error" not in html_result
