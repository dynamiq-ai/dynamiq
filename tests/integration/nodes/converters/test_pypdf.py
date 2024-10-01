from io import BytesIO

from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.nodes.converters.pypdf import PyPDFConverter
from dynamiq.runnables import RunnableResult, RunnableStatus
from dynamiq.types import Document


def test_workflow_with_pypdf_converter():
    wf_pypdf = Workflow(
        flow=Flow(nodes=[PyPDFConverter()]),
    )
    file = BytesIO(
        b"%PDF-1.7\n\n1 0 obj  % entry point\n<<\n  /Type /Catalog\n  /Pages 2 0 R\n>>\nendobj\n\n2 0 obj\n<<\n  "
        b"/Type /Pages\n  /MediaBox [ 0 0 200 200 ]\n  /Count 1\n  /Kids [ 3 0 R ]\n>>\nendobj\n\n3 0 obj\n<<\n  "
        b"/Type /Page\n  /Parent 2 0 R\n  /Resources <<\n    /Font <<\n      /F1 4 0 R \n    >>\n  >>\n  /Contents "
        b"5 0 R\n>>\nendobj\n\n4 0 obj\n<<\n  /Type /Font\n  /Subtype /Type1\n  /BaseFont /Times-Roman\n>>\nendobj\n\n"
        b"5 0 obj  % page content\n<<\n  /Length 44\n>>\nstream\nBT\n70 50 TD\n/F1 12 Tf\n(Hello, world!) "
        b"Tj\nET\nendstream\nendobj\n\nxref\n0 6\n0000000000 65535 f \n0000000010 00000 n \n0000000079 00000 n "
        b"\n0000000173 00000 n \n0000000301 00000 n \n0000000380 00000 n \ntrailer\n<<\n  /Size 6\n  /Root 1 0 "
        b"R\n>>\nstartxref\n492\n%%EOF"
    )
    file.name = "mock.pdf"
    input_data = {"files": [file]}
    response = wf_pypdf.run(input_data=input_data)
    document_id = response.output[next(iter(response.output))]["output"]["documents"][0]["id"]
    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output={"documents": [Document(id=document_id, content="Hello, world!", metadata={"file_path": file.name})]},
    ).to_dict(skip_format_types={BytesIO, bytes})

    expected_output = {wf_pypdf.flow.nodes[0].id: expected_result}
    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output=expected_output,
    )
