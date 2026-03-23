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
Search Gmail messages with flexible filters.

### Parameters
- `from_email` (str, optional): Filter by sender's email address.
- `to_email` (str, optional): Filter by recipient's email address.
- `subject_contains` (str, optional): Filter by subject content.
- `has_attachment` (str, optional): Filter for emails with attachments.
- `is_unread` (str, optional): Filter for unread emails.
- `after` (str, optional): Filter emails sent after a specific date (YYYY/MM/DD).
- `before` (str, optional): Filter emails sent before a specific date (YYYY/MM/DD).
- `label_ids` (list[str], optional): List of Gmail label IDs to filter by.
- `query` (str, optional): Phrase to search for in emails (Gmail search syntax).
- `max_results` (int, optional): Maximum number of emails to return.
"""


class SearchEmailsInputSchema(BaseModel):
    from_email: str | None = Field(None, description="Email address of the sender.")
    to_email: str | None = Field(None, description="Email address of the recipient.")
    subject_contains: str | None = Field(None, description="Text to search in subject line.")
    has_attachment: bool | None = Field(None, description="Whether email must have attachments.")
    is_unread: bool | None = Field(None, description="Whether to include only unread emails.")
    after: str | None = Field(None, description="Filter emails after a specific date (YYYY/MM/DD).")
    before: str | None = Field(None, description="Filter emails before a specific date (YYYY/MM/DD).")
    label_ids: list[str] | None = Field(None, description="List of Gmail label IDs to filter by.")
    query: str | None = Field(None, description="Search query string, supports Gmail search operators.")
    max_results: int | None = Field(10, description="Maximum number of emails to return.")


class SearchEmails(GmailBase):
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "search-emails"
    description: str = DESCRIPTION_SEARCH_EMAILS
    input_schema: ClassVar[type[BaseModel]] = SearchEmailsInputSchema
    connection: GoogleOAuth2 = GoogleOAuth2()

    def _build_query(self, input_data: SearchEmailsInputSchema) -> str:
        """
        Builds the Gmail API search query.

        Args:
            input_data (SearchEmailsInputSchema): Input with the metadata.

        Returns:
            str: Combined Gmail search query string.
        """
        q_parts = []
        if input_data.from_email:
            q_parts.append(f"from:{input_data.from_email}")
        if input_data.to_email:
            q_parts.append(f"to:{input_data.to_email}")
        if input_data.subject_contains:
            q_parts.append(f"subject:{input_data.subject_contains}")
        if input_data.has_attachment:
            q_parts.append("has:attachment")
        if input_data.is_unread:
            q_parts.append("is:unread")
        if input_data.after:
            q_parts.append(f"after:{input_data.after}")
        if input_data.before:
            q_parts.append(f"before:{input_data.before}")
        if input_data.label_ids:
            q_parts.extend(f"label:{label}" for label in input_data.label_ids)
        if input_data.query:
            q_parts.append(input_data.query)
        return " ".join(q_parts).strip()

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
        self, input_data: SearchEmailsInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Search emails using Gmail API with combined filters.

        Args:
            input_data (SearchEmailsInputSchema): Search filters.
            config (RunnableConfig | None): Optional runtime configuration.
            **kwargs: Additional arguments.

        Returns:
            dict[str, Any]: Success status and list of email details.

        Raises:
            ToolExecutionException: If search fails.
        """
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        try:
            query = self._build_query(input_data)
            response = (
                self.client.users()
                .messages()
                .list(userId="me", q=query if query else None, maxResults=input_data.max_results)
                .execute()
            )

            messages = response.get("messages", [])
            result = [self._fetch_message_details(msg["id"]) for msg in messages]

            logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")
            return {"content": result}

        except Exception as e:
            raise ToolExecutionException(str(e), recoverable=True)

    def close(self):
        self.client.close()
