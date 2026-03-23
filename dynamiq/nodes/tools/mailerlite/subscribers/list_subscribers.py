from typing import ClassVar

from pydantic import BaseModel, Field

from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

from ..base import BaseMailerLite


class ListSubscribersInputSchema(BaseModel):
    """
    Input schema for listing  subscribers.
    """

    status_filter: str | None = Field(
        default=None,
        alias="filter[status]",
        description="Filter by status: 'active', 'unsubscribed', 'unconfirmed', 'bounced', 'junk'.",
    )
    limit: int | None = Field(
        default=25,
        description="Number of subscribers to retrieve (max is not explicitly stated in docs, default=25).",
    )
    cursor: str | None = Field(
        default=None,
        description="Cursor to retrieve a specific page of results.",
    )
    include_groups: bool = Field(
        default=False,
        alias="include_groups",
        description="Whether to include 'groups' info in the response. Will set 'include=groups'.",
    )


class ListSubscribers(BaseMailerLite):
    """
    Node to list  subscribers.
    """

    name: str = "ListSubscribers"
    description: str = "List all subscribers with optional filters."
    # If you want automatic validation, set the input schema
    input_schema: ClassVar[type[ListSubscribersInputSchema]] = ListSubscribersInputSchema

    def execute(self, input_data: ListSubscribersInputSchema, config: RunnableConfig | None = None, **kwargs):
        """
        Execute the node to list subscribers.
        """
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        api_url = self.connection.url + "subscribers"

        params = {}
        if input_data.status_filter:
            params["filter[status]"] = input_data.status_filter
        if input_data.limit is not None:
            params["limit"] = input_data.limit
        if input_data.cursor:
            params["cursor"] = input_data.cursor
        if input_data.include_groups:
            params["include"] = "groups"

        try:
            response = self.client.get(
                api_url,
                headers=self.connection.headers,
                params=params,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to list subscribers. Error: {e}")
            raise ToolExecutionException(
                f"Tool '{self.name}' failed to list subscribers. Error: {str(e)}.",
                recoverable=True,
            )

        return {"content": data}
