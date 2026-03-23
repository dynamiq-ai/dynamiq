import base64
import mimetypes
import os
from email.message import EmailMessage
from io import BytesIO
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.connections import GoogleOAuth2
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.tools.gmail.gmail_base import GmailBase
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

DESCRIPTION_CREATE_DRAFT = """## Create Draft Tool
Creates a draft email in Gmail.
Allows adding optional file attachments by providing file paths or file contents.

### Parameters
- `to` (str): Recipient email address.
- `subject` (str, optional): Email subject.
- `body` (str, optional): Email body content.
- `file_paths` (list[str], optional): List of file paths for files to attach to the draft. If provided,
each path should match an item in `files`.
- `files` (list[BytesIO | bytes], optional): List of file contents as BytesIO or bytes objects, used as attachments.
"""


class CreateDraftInputSchema(BaseModel):
    to: str = Field(..., description="Recipient email address.")
    subject: str | None = Field(None, description="Email subject.")
    body: str | None = Field(None, description="Email body content.")
    file_paths: list[str] | None = Field(
        default=None, description="List of file paths corresponding to each file in `files`."
    )
    files: list[BytesIO | bytes] | None = Field(default=None, description="List of files as BytesIO or bytes objects.")

    model_config = {"arbitrary_types_allowed": True}


class CreateDraft(GmailBase):
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "create-draft"
    description: str = DESCRIPTION_CREATE_DRAFT
    input_schema: ClassVar[type[BaseModel]] = CreateDraftInputSchema
    connection: GoogleOAuth2 = GoogleOAuth2()

    def execute(
        self, input_data: CreateDraftInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Creates a draft email.

        Args:
            input_data (CreateDraftInputSchema): Email details.
            config (RunnableConfig | None): Optional runtime config.
            **kwargs: Additional arguments.

        Returns:
            dict[str, Any]: Success status and draft details.

        Raises:
            ToolExecutionException: If draft creation fails.
        """
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        if input_data.body is None:
            input_data.body = ""

        try:
            message = EmailMessage()
            message.set_content(input_data.body)
            message["To"] = input_data.to
            message["Subject"] = input_data.subject

            # Handle attachments if provided
            if input_data.files is not None and input_data.file_paths is not None:
                for filename, file_obj in zip(input_data.file_paths, input_data.files):
                    filename = os.path.basename(filename)

                    file_obj.seek(0)
                    file_data = file_obj.read()
                    mime_type, _ = mimetypes.guess_type(filename)
                    maintype, subtype = (mime_type or "application/octet-stream").split("/", 1)
                    message.add_attachment(file_data, maintype=maintype, subtype=subtype, filename=filename)
            elif input_data.files or input_data.file_paths:
                raise ToolExecutionException(
                    "Both 'files' and 'file_paths' must be provided together.", recoverable=True
                )

            # Encode the message
            raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
            result = self.client.users().drafts().create(userId="me", body={"message": {"raw": raw}}).execute()

            logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")
            return {"content": result}

        except Exception as e:
            raise ToolExecutionException(str(e), recoverable=True)

    def close(self):
        self.client.close()
