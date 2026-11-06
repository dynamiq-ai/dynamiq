import base64
from io import BytesIO

import pytest
from reportlab.pdfgen import canvas

from dynamiq.connections import Anthropic as AnthropicConnection
from dynamiq.nodes.llms import Anthropic
from dynamiq.prompts import (
    Prompt,
    VisionMessage,
    VisionMessageFileContent,
    VisionMessageFileData,
    VisionMessageImageContent,
    VisionMessageImageURL,
    VisionMessageTextContent,
)
from dynamiq.runnables import RunnableStatus


def create_claude_llm(model):
    connection = AnthropicConnection()
    return Anthropic(
        connection=connection,
        model=model,
        max_tokens=1000,
        temperature=0,
    )


@pytest.mark.integration
@pytest.mark.parametrize("model", ["claude-sonnet-4-5", "claude-opus-4-5", "claude-opus-4-6"])
def test_anthropic_llm_with_base64_pdf_file(model):
    """Test Anthropic LLM with base64 PDF file input."""
    pdf_bytes = BytesIO()
    c = canvas.Canvas(pdf_bytes)
    c.drawString(100, 750, "Test PDF Content")
    c.save()
    pdf_bytes.seek(0)

    base64_pdf = base64.b64encode(pdf_bytes.getvalue()).decode("utf-8")
    pdf_data_url = f"data:application/pdf;base64,{base64_pdf}"

    llm = create_claude_llm(model)

    prompt = Prompt(
        messages=[
            VisionMessage(
                role="user",
                content=[
                    VisionMessageFileContent(
                        file=VisionMessageFileData(file_data=pdf_data_url),
                    ),
                    VisionMessageTextContent(text="What is in this file?"),
                ],
            )
        ]
    )

    result = llm.run(input_data={}, prompt=prompt)

    assert result.status == RunnableStatus.SUCCESS
    assert result.output is not None
    assert "content" in result.output
    assert isinstance(result.output["content"], str)
    assert len(result.output["content"]) > 0


@pytest.mark.integration
@pytest.mark.parametrize("model", ["claude-sonnet-4-5", "claude-opus-4-5", "claude-opus-4-6"])
def test_anthropic_llm_with_base64_pdf_image(model):
    """Test Anthropic LLM with base64 PDF file input."""
    pdf_bytes = BytesIO()
    c = canvas.Canvas(pdf_bytes)
    c.drawString(100, 750, "Test PDF Content")
    c.save()
    pdf_bytes.seek(0)

    base64_pdf = base64.b64encode(pdf_bytes.getvalue()).decode("utf-8")
    pdf_data_url = f"data:application/pdf;base64,{base64_pdf}"

    llm = create_claude_llm(model)

    prompt = Prompt(
        messages=[
            VisionMessage(
                role="user",
                content=[
                    VisionMessageImageContent(
                        image_url=VisionMessageImageURL(url=pdf_data_url),
                    ),
                    VisionMessageTextContent(text="What is in this file?"),
                ],
            )
        ]
    )

    result = llm.run(input_data={}, prompt=prompt)

    assert result.status == RunnableStatus.SUCCESS
    assert result.output is not None
    assert "content" in result.output
    assert isinstance(result.output["content"], str)
    assert len(result.output["content"]) > 0
