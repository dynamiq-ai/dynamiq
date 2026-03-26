import pytest
from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.runnables import RunnableResult, RunnableStatus

from dynamiq.nodes.transformers import HTMLMarkdownToText


@pytest.mark.parametrize(
    "value, result",
    [
        ("<h1>Main Title</h1><p>Welcome!</p>", "Main Title Welcome!"),
        ("# Markdown Header\n\nSome **bold** text.", "Markdown Header Some bold text."),
        ("<ul><li>Item A</li><li>Item B</li></ul>", "Item A Item B"),
        ("**Bold** and *italic* text in markdown.", "Bold and italic text in markdown."),
        ("<script>alert('XSS');</script><p>Valid content</p>", "Valid content"),
        ("", ""),
        ("<p>     Trim   spaces   </p>", "Trim spaces"),
    ],
)
def test_workflow_with_html_markdown_to_text(value, result):
    wf_html_markdown_to_text = Workflow(flow=Flow(nodes=[HTMLMarkdownToText()]))

    input_data = {"value": value}
    output = {"content": result}
    response = wf_html_markdown_to_text.run(input_data=input_data)

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output=output,
    ).to_dict()

    expected_output = {wf_html_markdown_to_text.flow.nodes[0].id: expected_result}
    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output=expected_output,
    )
