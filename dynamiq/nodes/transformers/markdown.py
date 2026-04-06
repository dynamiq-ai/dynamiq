import base64
import html
from io import BytesIO
from typing import Any, ClassVar, Literal

import markdown
from bs4 import BeautifulSoup
from inscriptis import ParserConfig, get_text
from pydantic import BaseModel, ConfigDict, Field
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.node import ensure_config
from dynamiq.runnables import RunnableConfig


class HTMLMarkdownToTextInputSchema(BaseModel):
    value: str = Field(..., description="Parameter to provide value to transform")


class HTMLMarkdownToText(Node):
    group: Literal[NodeGroup.TRANSFORMERS] = NodeGroup.TRANSFORMERS
    name: str = "html-markdown-to-text"
    description: str = "Node that returns converts HTML/Markdown to string format"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_schema: ClassVar[type[HTMLMarkdownToTextInputSchema]] = HTMLMarkdownToTextInputSchema

    def execute(
        self, input_data: HTMLMarkdownToTextInputSchema, config: RunnableConfig = None, **kwargs
    ) -> dict[str, Any]:
        """
        Converts Markdown and HTML content into plain text.

        Args:
            input_data (HTMLMarkdownToTextInputSchema): input data for the tool, which includes value to transform.
            config (RunnableConfig, optional): Configuration for the runnable, including callbacks.
            **kwargs: Additional arguments passed to the execution context.

        Returns:
            dict[str, Any]: A dictionary containing transformed text.
        """
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        value = input_data.value
        try:
            if bool(BeautifulSoup(value, "html.parser").find()):
                soup = BeautifulSoup(value, "html.parser")
            else:
                value = markdown.markdown(value)
                soup = BeautifulSoup(value, "html.parser")

            text = soup.get_text(separator=" ")
            text = html.unescape(text)
            text = " ".join(text.split())
            return {"content": text}
        except Exception as e:
            raise ValueError(f"Encountered an error while performing transforming. \nError details: {e}")


class MarkdownToPDFInputSchema(BaseModel):
    value: str = Field(..., description="Parameter to provide markdown for transformation")


class MarkdownToPDF(Node):
    group: Literal[NodeGroup.TRANSFORMERS] = NodeGroup.TRANSFORMERS
    name: str = "markdown-to-pdf"
    description: str = "Node that transforms markdown into a pdf data."
    cursor_x: int = 100
    cursor_y: int = 750

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_schema: ClassVar[type[MarkdownToPDFInputSchema]] = MarkdownToPDFInputSchema

    def execute(self, input_data: MarkdownToPDFInputSchema, config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        """
        Transforms markdown into a pdf, encoded into base64 format.

        Args:
            input_data (RegexExtractionInputSchema): input data for the tool, which includes value to transform and
                pattern to use.
            config (RunnableConfig, optional): Configuration for the runnable, including callbacks.
            **kwargs: Additional arguments passed to the execution context.

        Returns:
            dict[str, Any]: A dictionary containing data in base64 format.
        """
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        value = input_data.value
        try:
            value = markdown.markdown(value)
            text = get_text(value, ParserConfig(display_links=True))

            buffer = BytesIO()
            c = canvas.Canvas(buffer, pagesize=letter)
            text_object = c.beginText(self.cursor_x, self.cursor_y)
            lines = text.split("\n")
            for line in lines:
                text_object.textLine(line)
            c.drawText(text_object)
            c.showPage()
            c.save()
            buffer.seek(0)

            pdf_bytes = buffer.read()
            pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")
            return {"content": pdf_base64}
        except Exception as e:
            raise ValueError(f"Encountered an error while performing transformation. \nError details: {e}")
