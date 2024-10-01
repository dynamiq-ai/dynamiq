from io import BytesIO

from pptx import Presentation

from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.nodes.converters.pptx import PPTXFileConverter
from dynamiq.runnables import RunnableResult, RunnableStatus
from dynamiq.types import Document


def test_workflow_with_pptx_converter():
    content = "Hello, World!"
    prs = Presentation()
    title_slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(title_slide_layout)
    title = slide.shapes.title
    title.text = content

    pptx_converter = PPTXFileConverter()
    wf_pptx = Workflow(flow=Flow(nodes=[pptx_converter]))
    file = BytesIO()
    prs.save(file)
    file.name = "mock.pptx"
    input_data = {"files": [file]}

    response = wf_pptx.run(input_data=input_data)
    document_id = response.output[next(iter(response.output))]["output"]["documents"][0]["id"]
    pptx_converter_expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output={"documents": [Document(id=document_id, content=content, metadata={"file_path": file.name})]},
    ).to_dict(skip_format_types={BytesIO, bytes})

    expected_output = {pptx_converter.id: pptx_converter_expected_result}
    assert response == RunnableResult(status=RunnableStatus.SUCCESS, input=input_data, output=expected_output)
