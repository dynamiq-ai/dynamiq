from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.connections import GoogleOAuth2
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.tools.google_docs.google_docs_base import GoogleDocsBase
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

DESCRIPTION_READ_DOC = """## Reads Google Document
Retrieves the content of a Google Docs document by its ID.

### Parameters
- `document_id` (str): ID of the document to read.
"""


class ReadDocumentInputSchema(BaseModel):
    document_id: str = Field(..., description="ID of the document to read.")


class ReadDocument(GoogleDocsBase):
    """Reads the content of a Google Docs document by its ID."""

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "read-google-doc"
    description: str = DESCRIPTION_READ_DOC
    input_schema: ClassVar[type[BaseModel]] = ReadDocumentInputSchema
    connection: GoogleOAuth2 = GoogleOAuth2()

    def execute(
        self, input_data: ReadDocumentInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """Retrieves the full content of a Google Docs document.
        Args:
            input_data (ReadDocumentInputSchema): Document parameters.
            config (RunnableConfig | None): Optional runtime configuration.
            **kwargs: Additional arguments.

        Returns:
            dict[str, Any]: The update result details.

        Raises:
            ToolExecutionException: If update fails or document_id is missing.
        """
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        document_id = input_data.document_id

        try:
            logger.info(f"{self.name} ({self.id}) - Reading document ID '{document_id}'")
            document = self.client.documents().get(documentId=document_id).execute()
            body_content = document.get("body", {}).get("content", [])

            text_parts = []
            for element in body_content:
                if "paragraph" in element:
                    for p_elem in element["paragraph"].get("elements", []):
                        text_parts.append(p_elem.get("textRun", {}).get("content", ""))

            full_text = "".join(text_parts)
            result = {
                "content": full_text,
                "title": document.get("title"),
            }

            logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")
            return {"content": result}

        except Exception as e:
            raise ToolExecutionException(f"Failed to read document: {e}", recoverable=True)

    def close(self):
        """Closes the underlying Google Docs client."""
        self.client.close()
