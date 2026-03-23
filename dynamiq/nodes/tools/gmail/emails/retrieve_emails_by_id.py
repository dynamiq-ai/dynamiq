from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.connections import GoogleOAuth2
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.tools.gmail.emails.utils import extract_body, extract_headers
from dynamiq.nodes.tools.gmail.gmail_base import GmailBase
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

DESCRIPTION_SEARCH_EMAILS = """## Search Emails Tool
Retrieve Gmail emails and threads by specific IDs.

### Parameters
- `email_ids` (list[str], optional): List of specific email IDs to retrieve.
- `thread_ids` (list[str], optional): List of specific thread IDs to retrieve all emails from.
"""


class RetrieveEmailsByIdInputSchema(BaseModel):
    email_ids: list[str] | None = Field(None, description="List of specific email IDs to retrieve.")
    thread_ids: list[str] | None = Field(None, description="List of specific thread IDs to retrieve.")


class RetrieveEmailsById(GmailBase):
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "retrieve-emails-by-id"
    description: str = DESCRIPTION_SEARCH_EMAILS
    input_schema: ClassVar[type[BaseModel]] = RetrieveEmailsByIdInputSchema
    connection: GoogleOAuth2 = GoogleOAuth2()

    def _fetch_message_details(self, msg_id: str) -> dict[str, Any]:
        """
        Fetches full message details by message ID from Gmail API and extracts useful info.

        Args:
            msg_id (str): The Gmail message ID.

        Returns:
            dict[str, Any]: Dictionary containing message id, thread id, labels, snippet,
                            subject, from, to, date, and plain text body.
        """
        msg_data = self.client.users().messages().get(userId="me", id=msg_id, format="full").execute()
        headers = extract_headers(msg_data)
        body = extract_body(msg_data)
        return {
            "id": msg_id,
            "threadId": msg_data.get("threadId"),
            "labelIds": msg_data.get("labelIds", []),
            "snippet": msg_data.get("snippet", ""),
            "subject": headers.get("Subject", ""),
            "from": headers.get("From", ""),
            "to": headers.get("To", ""),
            "date": headers.get("Date", ""),
            "body": body,
        }

    def execute(
        self, input_data: RetrieveEmailsByIdInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        try:
            emails = []
            threads = []

            for msg_id in input_data.email_ids:
                email = self._fetch_message_details(msg_id)
                emails.append({msg_id: email})

            for thread_id in input_data.thread_ids:
                thread = self.client.users().threads().get(userId="me", id=thread_id, format="full").execute()
                thread_emails = []
                for msg in thread.get("messages", []):
                    email = self._fetch_message_details(msg["id"])
                    thread_emails.append(email)
                threads.append({thread_id: thread_emails})

            result = {"success": True, "emails": emails, "threads": threads}

            logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")
            return {"content": result}

        except Exception as e:
            raise ToolExecutionException(str(e), recoverable=True)

    def close(self):
        self.client.close()
