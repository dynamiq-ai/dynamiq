from io import BytesIO

import pytest
from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.runnables import RunnableResult, RunnableStatus

from dynamiq.nodes.transformers import TextTemplate
from dynamiq.nodes.transformers.text import TextTemplateInputSchema


@pytest.mark.parametrize(
    "template, input_data, result",
    [
        (
            "Hello, my name is {{name}} and I am {{age}} years old.",
            {"info": "and I live in NY", "template": "Hello, my name is Alice {{info}}."},
            "Hello, my name is Alice and I live in NY.",
        ),
        ("My name is {{name}}.", {"name": "Bob"}, "My name is Bob."),
        ("This is a regular text.", {}, "This is a regular text."),
        (
            "{{greeting}} {{name}}! Welcome to {{place}}.",
            {"greeting": "Hello", "name": "Charlie", "place": "Paris"},
            "Hello Charlie! Welcome to Paris.",
        ),
        ("Price: {{price}}$", {"price": "100"}, "Price: 100$"),
        ("Are you a student? {{is_student}}.", {"is_student": "Yes"}, "Are you a student? Yes."),
        (
            "{{title}}: {{content}}",
            {"title": "Introduction", "content": "This is a long paragraph that contains a lot of content."},
            "Introduction: This is a long paragraph that contains a lot of content.",
        ),
    ],
)
def test_workflow_with_text_template_render(template, input_data, result):
    wf_text_merge = Workflow(flow=Flow(nodes=[TextTemplate(template=template)]))

    output = {"content": result}
    response = wf_text_merge.run(input_data=input_data)

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=dict(TextTemplateInputSchema(**input_data)),
        output=output,
    ).to_dict(skip_format_types={BytesIO})

    expected_output = {wf_text_merge.flow.nodes[0].id: expected_result}
    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output=expected_output,
    )
