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

DESCRIPTION_SEND_EMAIL = """## Send Email Tool
### Description
Sends an email using the Gmail API.

### Parameters
- `to` (str): Recipient email address.
- `subject` (str, optional): Subject of the email.
- `body` (str, optional): The plain text content of the email.
- `file_paths` (list[str], optional): Parameter to provide file paths corresponding to the files list.
- `files` (list[BytesIO | bytes], optional): Parameter to provide file contents as BytesIO or bytes objects.
"""


class SendEmailInputSchema(BaseModel):
    to: str = Field(..., description="Recipient email address")
    subject: str | None = Field(default=None, description="Subject line of the email (can be empty)")
    body: str | None = Field(default=None, description="Plain text content of the email (can be empty)")
    file_paths: list[str] | None = Field(
        default=None, description="List of file paths corresponding to each file in `files`."
    )
    files: list[BytesIO | bytes] | None = Field(default=None, description="List of files as BytesIO or bytes objects.")

    model_config = {"arbitrary_types_allowed": True}


class SendEmail(GmailBase):
    """
    Tool for sending emails via Gmail.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "send-email"
    description: str = DESCRIPTION_SEND_EMAIL
    input_schema: ClassVar[type[SendEmailInputSchema]] = SendEmailInputSchema
    connection: GoogleOAuth2 = GoogleOAuth2()

    def execute(
        self, input_data: SendEmailInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Executes the email sending operation using the Gmail API.

        Args:
            input_data (SendEmailInputSchema): The input data containing the recipient email,
                subject, and body of the message.
            config (RunnableConfig | None): Optional execution configuration containing
                callback hooks and metadata.
            **kwargs: Additional keyword arguments passed to the node execution lifecycle.

        Returns:
            dict[str, Any]: A dictionary containing the success status and Gmail API response details.

        Raises:
            ToolExecutionException: Raised when an error occurs during the email sending process.
        """
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        if input_data.body is None:
            input_data.body = ""

        try:
            result = self._send_email(input_data)

            logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")
            return {"content": result}
        except Exception as exc:
            logger.error(f"{self.name} ({self.id}) - Failed to send email: {exc}")
            raise ToolExecutionException(f"Failed to send email: {exc}", recoverable=False)

    def _send_email(self, input_data: SendEmailInputSchema) -> dict[str, Any]:
        """
        Builds and sends the email via Gmail API.

        Args:
            input_data (SendEmailInputSchema): Email data including recipient, subject, body, and optional attachments.

        Returns:
            dict[str, Any]: The Gmail API response after sending the message.
        """
        message = EmailMessage()
        message.set_content(input_data.body)
        message["To"] = input_data.to
        message["From"] = "me"
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
            raise ToolExecutionException("Both 'files' and 'file_paths' must be provided together.", recoverable=True)

        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        request_body = {"raw": encoded_message}

        return self.client.users().messages().send(userId="me", body=request_body).execute()

    def close(self):
        self.client.close()
