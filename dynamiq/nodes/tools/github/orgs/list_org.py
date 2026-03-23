from typing import ClassVar

from pydantic import BaseModel, Field

from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

from ..base import BaseGitHub


class ListOrganizationsInputSchema(BaseModel):
    """
    Input parameters for listing GitHub organizations.
    """

    since: int | None = Field(
        default=None, description="An organization ID. Only return orgs with an ID greater than this ID."
    )
    per_page: int | None = Field(default=None, description="The number of results per page (max 100). Default is 30.")


class ListOrganizations(BaseGitHub):
    """
    Calls GET /organizations to list organizations in GitHub.
    """

    name: str = "GitHubListOrganizations"
    description: str = "List organizations from GitHub."
    input_schema: ClassVar[type[ListOrganizationsInputSchema]] = ListOrganizationsInputSchema

    def execute(self, input_data: ListOrganizationsInputSchema, config: RunnableConfig | None = None, **kwargs):
        logger.info(f"Tool {self.name} - {self.id}: started with input: {input_data.dict()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        params = {}
        if input_data.since is not None:
            params["since"] = input_data.since
        if input_data.per_page is not None:
            params["per_page"] = input_data.per_page

        api_url = f"{self.connection.url}/organizations"

        try:
            response = self.client.get(api_url, headers=self.connection.headers, params=params)
            response.raise_for_status()
            orgs_list = response.json()
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to list organizations. Error: {e}")
            raise ToolExecutionException(
                f"Tool '{self.name}' failed to list organizations. Error: {str(e)}",
                recoverable=True,
            )

        if self.is_optimized_for_agents:
            summary = []
            for org in orgs_list:
                summary.append(f"- {org.get('login', 'unknown')} (ID: {org.get('id', 'n/a')})")
            return {"content": "Organizations:\n" + "\n".join(summary)}
        else:
            return {"content": orgs_list}
