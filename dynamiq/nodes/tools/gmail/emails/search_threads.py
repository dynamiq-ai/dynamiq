from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.connections import GoogleOAuth2
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.tools.gmail.gmail_base import GmailBase
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

DESCRIPTION_SEARCH_EMAILS = """## Search Threads Tool
Search Gmail threads with flexible filters.

### Parameters
- `from_email` (str, optional): Filter by sender's email address.
- `to_email` (str, optional): Filter by recipient's email address.
- `subject_contains` (str, optional): Filter by subject content.
- `has_attachment` (str, optional): Filter for threads with attachments.
- `is_unread` (str, optional): Filter for unread threads.
- `after` (str, optional): Filter threads sent after a specific date (YYYY/MM/DD).
- `before` (str, optional): Filter threads sent before a specific date (YYYY/MM/DD).
- `label_ids` (list[str], optional): List of Gmail label IDs to filter by.
- `query` (str, optional): Free-form Gmail search query.
- `max_results` (int, optional): Maximum number of threads to return.
"""


class SearchThreadsInputSchema(BaseModel):
    from_email: str | None = Field(None, description="Email address of the sender.")
    to_email: str | None = Field(None, description="Email address of the recipient.")
    subject_contains: str | None = Field(None, description="Text to search in subject line.")
    has_attachment: bool | None = Field(None, description="Whether thread must have attachments.")
    is_unread: bool | None = Field(None, description="Whether to include only unread threads.")
    after: str | None = Field(None, description="Filter threads after a specific date (YYYY/MM/DD).")
    before: str | None = Field(None, description="Filter threads before a specific date (YYYY/MM/DD).")
    label_ids: list[str] | None = Field(None, description="List of label IDs to filter threads.")
    query: str | None = Field(None, description="Free-form Gmail search query.")
    max_results: int = Field(default=10, description="Maximum number of threads to return.")


class SearchThreads(GmailBase):
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "search-threads"
    description: str = DESCRIPTION_SEARCH_EMAILS
    input_schema: ClassVar[type[BaseModel]] = SearchThreadsInputSchema
    connection: GoogleOAuth2 = GoogleOAuth2()

    def execute(
        self, input_data: SearchThreadsInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Searches Gmail threads using structured filter fields.

        Returns:
            dict[str, Any]: Success status and list of matching threads.

        Raises:
            ToolExecutionException: If the search fails.
        """
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        try:
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
                for label in input_data.label_ids:
                    q_parts.append(f"label:{label}")
            if input_data.query:
                q_parts.append(input_data.query)

            query = " ".join(q_parts).strip()

            response = (
                self.client.users()
                .threads()
                .list(userId="me", q=query if query else None, maxResults=input_data.max_results)
                .execute()
            )

            result = response.get("threads", [])

            logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")
            return {"content": result}

        except Exception as e:
            raise ToolExecutionException(str(e), recoverable=True)

    def close(self):
        self.client.close()
