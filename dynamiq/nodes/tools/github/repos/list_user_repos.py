from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

from ..base import BaseGitHub


class ListUserRepositoriesInputSchema(BaseModel):
    """
    Input parameters for listing repositories for a given user.

    Docs: https://docs.github.com/en/rest/repos/repos#list-repositories-for-a-user
    """

    username: str = Field(..., description="The GitHub username to list repositories for.")
    type: Literal["all", "owner", "member"] | None = Field(
        default="owner", description="Filter repos by type. Defaults to 'owner'."
    )
    sort: Literal["created", "updated", "pushed", "full_name"] | None = Field(
        default="full_name", description="The property to sort by. Default is 'full_name'."
    )
    direction: Literal["asc", "desc"] | None = Field(
        default=None, description="The order to sort by. Default: 'asc' if sort=full_name, otherwise 'desc'."
    )
    per_page: int | None = Field(default=None, description="Number of results per page (max 100). Default is 30.")
    page: int | None = Field(default=None, description="Page number of the results to fetch. Default is 1.")


class ListUserRepositories(BaseGitHub):
    """
    Calls GET /users/{username}/repos to list repositories for a user.
    """

    name: str = "GitHubListUserRepositories"
    description: str = "List public repositories for a specified user (by default) or all if token permits."
    input_schema: ClassVar[type[ListUserRepositoriesInputSchema]] = ListUserRepositoriesInputSchema

    def execute(self, input_data: ListUserRepositoriesInputSchema, config: RunnableConfig | None = None, **kwargs):
        logger.info(f"Node {self.name} - {self.id}: started with input {input_data.dict()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        api_url = f"{self.connection.url}/users/{input_data.username}/repos"
        params = {}
        if input_data.type:
            params["type"] = input_data.type
        if input_data.sort:
            params["sort"] = input_data.sort
        if input_data.direction:
            params["direction"] = input_data.direction
        if input_data.per_page:
            params["per_page"] = input_data.per_page
        if input_data.page:
            params["page"] = input_data.page

        try:
            response = self.client.get(api_url, headers=self.connection.headers, params=params)
            response.raise_for_status()
            repos_list = response.json()
        except Exception as e:
            logger.error(f"Node {self.name} - {self.id}: failed to list user repos. Error: {e}")
            raise ToolExecutionException(
                f"Tool '{self.name}' failed to list repos for user '{input_data.username}'. Error: {str(e)}",
                recoverable=True,
            )

        if self.is_optimized_for_agents:
            lines = [f"- {repo.get('full_name')}" for repo in repos_list]
            return {"content": f"Repositories for user '{input_data.username}':\n" + "\n".join(lines)}
        else:
            return {"content": repos_list}
