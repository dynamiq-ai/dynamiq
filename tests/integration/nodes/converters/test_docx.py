from io import BytesIO

from docx import Document as DocxDocument

from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.nodes.converters.docx import DOCXFileConverter
from dynamiq.runnables import RunnableResult, RunnableStatus
from dynamiq.types import Document


def test_workflow_with_docx_converter():
    content = "Hello, World!"

    doc = DocxDocument()
    doc.add_paragraph(content)

    docx_converter = DOCXFileConverter()
    wf_docx = Workflow(flow=Flow(nodes=[docx_converter]))
    file = BytesIO()
    doc.save(file)
    file.name = "mock.docx"
    input_data = {"files": [file]}

    response = wf_docx.run(input_data=input_data)
    document_id = response.output[next(iter(response.output))]["output"]["documents"][0]["id"]
    docx_converter_expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output={"documents": [Document(id=document_id, content=content, metadata={"file_path": file.name})]},
    ).to_dict(skip_format_types={BytesIO, bytes})

    expected_output = {docx_converter.id: docx_converter_expected_result}
    assert response == RunnableResult(status=RunnableStatus.SUCCESS, input=input_data, output=expected_output)
