from typing import ClassVar

from pydantic import BaseModel, Field, conint

from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

from ..base import BaseMailerLite


class ListGroupsInputSchema(BaseModel):
    """
    Input schema for listing all groups with optional filters.
    """

    limit: conint(gt=0) | None = Field(default=None, description="Max number of groups to retrieve (1-1000).")
    page: conint(gt=0) | None = Field(default=None, description="Pagination page number (starts from 1).")
    filter_name: str | None = Field(default=None, alias="filter[name]", description="Partial match on group name.")
    sort: str | None = Field(
        default=None,
        description=(
            "Sort by any of these fields: 'name', 'total', 'open_rate', 'click_rate', 'created_at'. "
            "Use '-' prefix for descending (e.g., '-total')."
        ),
    )


class ListGroups(BaseMailerLite):
    """
    Node to list all groups in .
    """

    name: str = "ListGroups"
    description: str = "List all  groups with optional pagination and filters."
    input_schema: ClassVar[type[ListGroupsInputSchema]] = ListGroupsInputSchema

    def execute(self, input_data: ListGroupsInputSchema, config: RunnableConfig | None = None, **kwargs):
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        api_url = self.connection.url + "groups"

        params = {}
        if input_data.limit is not None:
            params["limit"] = input_data.limit
        if input_data.page is not None:
            params["page"] = input_data.page
        if input_data.filter_name:
            params["filter[name]"] = input_data.filter_name
        if input_data.sort:
            params["sort"] = input_data.sort

        try:
            response = self.client.get(
                api_url,
                headers=self.connection.headers,
                params=params,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"{self.name} - {self.id}: failed to list groups. Error: {e}")
            raise ToolExecutionException(
                f"Failed to list groups. Error: {str(e)}.",
                recoverable=True,
            )

        return {"content": data}
