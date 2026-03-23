from typing import ClassVar

from pydantic import BaseModel, Field

from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

from ..base import BaseGitHub


class ListUserOrganizationsInputSchema(BaseModel):
    """
    Input parameters for listing the public organizations of a given user.
    """

    username: str = Field(..., description="The handle (username) of the GitHub user.")
    per_page: int | None = Field(default=None, description="Number of results per page (max 100). Default is 30.")
    page: int | None = Field(default=None, description="Page number of the results to fetch. Default is 1.")


class ListUserOrganizations(BaseGitHub):
    """
    Calls GET /users/{username}/orgs to list public organizations for a specified user.
    """

    name: str = "GitHubListUserOrganizations"
    description: str = "List public organizations for a specified user in GitHub."
    input_schema: ClassVar[type[ListUserOrganizationsInputSchema]] = ListUserOrganizationsInputSchema

    def execute(self, input_data: ListUserOrganizationsInputSchema, config: RunnableConfig | None = None, **kwargs):
        logger.info(f"Tool {self.name} - {self.id}: started with input: {input_data.dict()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        api_url = f"{self.connection.url}/users/{input_data.username}/orgs"
        params = {}
        if input_data.per_page:
            params["per_page"] = input_data.per_page
        if input_data.page:
            params["page"] = input_data.page

        try:
            response = self.client.get(api_url, headers=self.connection.headers, params=params)
            response.raise_for_status()
            orgs_list = response.json()
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to list user's organizations. Error: {e}")
            raise ToolExecutionException(
                f"Tool '{self.name}' failed to list organizations for user {input_data.username}. Error: {str(e)}",
                recoverable=True,
            )

        if self.is_optimized_for_agents:
            lines = []
            for org in orgs_list:
                lines.append(f"- {org.get('login')} (ID: {org.get('id')})")
            return {"content": "Public Organizations:\n" + "\n".join(lines)}
        else:
            return {"content": orgs_list}
