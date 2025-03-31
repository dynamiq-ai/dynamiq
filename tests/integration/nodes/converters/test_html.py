from io import BytesIO

from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.nodes.converters.html import HTMLConverter
from dynamiq.runnables import RunnableStatus


def test_workflow_with_html_converter():
    wf_html = Workflow(
        flow=Flow(nodes=[HTMLConverter()]),
    )

    # Create a simple HTML file content
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

    # Verify the response status
    assert response.status == RunnableStatus.SUCCESS

    # Verify that documents were created
    node_id = wf_html.flow.nodes[0].id
    assert "documents" in response.output[node_id]["output"]
    assert len(response.output[node_id]["output"]["documents"]) == 1

    # Get the actual document content
    document = response.output[node_id]["output"]["documents"][0]

    # Check for expected elements in the content
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

    # Create a more complex HTML file with various elements
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

    # Verify the response status
    assert response.status == RunnableStatus.SUCCESS

    # Verify that documents were created
    node_id = wf_html.flow.nodes[0].id
    assert "documents" in response.output[node_id]["output"]
    assert len(response.output[node_id]["output"]["documents"]) == 1

    # Verify document content contains expected markdown elements
    document = response.output[node_id]["output"]["documents"][0]
    content = document["content"]

    # Check for various markdown elements in the content
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
