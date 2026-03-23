from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.connections import GoogleOAuth2
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.tools.google_docs.google_docs_base import GoogleDocsBase
from dynamiq.nodes.tools.google_docs.utils import markdown_to_requests
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

DESCRIPTION_UPDATE_DOC = """## Updates Google Document
Updates a Google Docs document by changing the title and/or the body content.

### Parameters
- `document_id` (str): ID of the document to update. Required to update an existing doc.
- `body` (str): Body text to insert.
- `parse_markdown` (bool, optional): Whether to parse the body as Markdown (default False).
"""


class UpdateDocumentInputSchema(BaseModel):
    document_id: str | None = Field(..., description="ID of the existing document to modify.")
    body: str | None = Field(..., description="Body content to update. If not provided, body won't be changed.")
    parse_markdown: bool = Field(default=False, description="Whether to parse the body as Markdown.")


class UpdateDocument(GoogleDocsBase):
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "update-google-doc"
    description: str = DESCRIPTION_UPDATE_DOC
    input_schema: ClassVar[type[BaseModel]] = UpdateDocumentInputSchema
    connection: GoogleOAuth2 = GoogleOAuth2()

    def execute(
        self, input_data: UpdateDocumentInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Updates a Google Docs document's content.

        Args:
            input_data (UpdateDocInputSchema): Document update parameters.
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
        if not document_id:
            raise ToolExecutionException("document_id is required to update a document.", recoverable=False)

        logger.info(f"{self.name} ({self.id}) - Updating document ID '{document_id}'")

        try:
            requests = []

            if input_data.body is not None:
                doc = self.client.documents().get(documentId=document_id).execute()
                body_content = doc.get("body", {}).get("content", [])
                if not body_content:
                    end_index = 1
                else:
                    end_index = body_content[-1].get("endIndex", 1)

                if end_index > 2:
                    requests.append({"deleteContentRange": {"range": {"startIndex": 1, "endIndex": end_index - 1}}})

                if input_data.parse_markdown:
                    requests += markdown_to_requests(input_data.body)
                else:
                    text = input_data.body + "\n"
                    requests.append({"insertText": {"location": {"index": 1}, "text": text}})

            if requests:
                result = (
                    self.client.documents().batchUpdate(documentId=document_id, body={"requests": requests}).execute()
                )
            else:
                result = {}

            logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")
            return {"content": result}

        except Exception as e:
            raise ToolExecutionException(f"Failed to update document: {e}", recoverable=True)

    def close(self):
        """Closes the client."""
        self.client.close()
