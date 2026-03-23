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

DESCRIPTION_CREATE_DOC = """## Creates Google Document
Creates a new Google Docs document with optional title and body content.

### Parameters
- `title` (str, optional): Title of the new document. Defaults to 'Untitled Document'.
- `body` (str, optional): Body content to insert.
- `parse_markdown` (bool, optional): Whether to parse the body as Markdown (default False).
"""


class CreateDocumentInputSchema(BaseModel):
    title: str | None = Field("Untitled Document", description="Title of the new document.")
    body: str | None = Field(None, description="Body content to insert.")
    parse_markdown: bool = Field(default=False, description="Whether to parse the body as Markdown.")


class CreateDocument(GoogleDocsBase):
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "create-google-doc"
    description: str = DESCRIPTION_CREATE_DOC
    input_schema: ClassVar[type[BaseModel]] = CreateDocumentInputSchema
    connection: GoogleOAuth2 = GoogleOAuth2()

    def execute(
        self, input_data: CreateDocumentInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Creates a new Google Docs document with an optional title and body content.

        Args:
            input_data (CreateDocInputSchema): Document creation parameters.
            config (RunnableConfig | None): Optional runtime configuration.
            **kwargs: Additional arguments.

        Returns:
            dict[str, Any]: The created document details including document ID.

        Raises:
            ToolExecutionException: If creation fails.
        """
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        title = input_data.title or "Untitled Document"

        try:
            doc_metadata = {"title": title}
            doc = self.client.documents().create(body=doc_metadata).execute()
            document_id = doc.get("documentId")
            logger.info(f"{self.name} ({self.id}) - Created document ID '{document_id}' with title '{title}'")

            requests = []

            if input_data.body:
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
            raise ToolExecutionException(f"Failed to create document: {e}", recoverable=True)

    def close(self):
        """Closes the client."""
        self.client.close()
