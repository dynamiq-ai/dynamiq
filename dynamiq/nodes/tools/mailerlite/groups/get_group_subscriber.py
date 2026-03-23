from typing import ClassVar

from pydantic import BaseModel, Field

from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

from ..base import BaseMailerLite


class GetGroupSubscribersInputSchema(BaseModel):
    """
    Input schema for retrieving subscribers for a group.
    """

    group_id: str = Field(..., description="The group ID.")
    status_filter: str | None = Field(
        default=None,
        alias="filter[status]",
        description="Filter by status: 'active', 'unsubscribed', 'unconfirmed', 'bounced', 'junk'.",
    )
    limit: int | None = Field(default=25, description="Number of subscribers to retrieve (default=25).")
    cursor: str | None = Field(default=None, description="Cursor for pagination.")
    include_groups: bool = Field(
        default=False,
        alias="include_groups",
        description="If True, sets 'include=groups' to retrieve group data in the subscriber object.",
    )


class GetGroupSubscribers(BaseMailerLite):
    """
    Node to retrieve subscribers for a specific group in .
    """

    name: str = "GetGroupSubscribers"
    description: str = "Get all subscribers for a given group in ."
    input_schema: ClassVar[type[GetGroupSubscribersInputSchema]] = GetGroupSubscribersInputSchema

    def execute(self, input_data: GetGroupSubscribersInputSchema, config: RunnableConfig | None = None, **kwargs):
        logger.info(f"Tool {self.name} - {self.id}: retrieving group subscribers:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        api_url = self.connection.url + f"groups/{input_data.group_id}/subscribers"

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
            logger.error(f"{self.name} - {self.id}: failed to get group subscribers. Error: {e}")
            raise ToolExecutionException(
                f"Failed to get subscribers for group '{input_data.group_id}'. Error: {str(e)}.",
                recoverable=True,
            )

        return {"content": data}
