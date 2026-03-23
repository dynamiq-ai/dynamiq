from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

from ..base import BaseGitHub


class ListPullRequestsInputSchema(BaseModel):
    """
    Input parameters for listing pull requests in a repository.
    """

    owner: str = Field(..., description="The account owner of the repository.")
    repo: str = Field(..., description="The name of the repository (without the .git extension).")
    state: Literal["open", "closed", "all"] | None = Field(
        default="open",
        description="Filter pull requests by state. One of: open, closed, all. Default: open.",
    )
    head: str | None = Field(
        default=None, description="Filter pulls by head user/organization + branch name. e.g. github:new-script-format"
    )
    base: str | None = Field(default=None, description="Filter pulls by base branch name, e.g. 'gh-pages'.")
    sort: Literal["created", "updated", "popularity", "long-running"] | None = Field(
        default="created",
        description=(
            "The field to sort by. 'popularity' sorts by number of comments. 'long-running' sorts by date created "
            "and limits results to pull requests that have been open for >1 month and had activity in past month."
        ),
    )
    direction: Literal["asc", "desc"] | None = Field(
        default=None,
        description="The direction of the sort. If sort=created or not specified, default is 'desc'; otherwise 'asc'.",
    )
    per_page: int | None = Field(default=None, description="Number of results per page (max 100). Default is 30.")
    page: int | None = Field(default=None, description="Page number of the results to fetch. Default is 1.")


class ListPullRequests(BaseGitHub):
    """
    Calls GET /repos/{owner}/{repo}/pulls to list pull requests in a repository.
    """

    name: str = "GitHubListPullRequests"
    description: str = "List pull requests in a given repository."
    input_schema: ClassVar[type[ListPullRequestsInputSchema]] = ListPullRequestsInputSchema

    def execute(self, input_data: ListPullRequestsInputSchema, config: RunnableConfig | None = None, **kwargs):
        logger.info(f"Tool {self.name} - {self.id}: started with input: {input_data.dict()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        api_url = f"{self.connection.url}/repos/{input_data.owner}/{input_data.repo}/pulls"

        params = {}
        if input_data.state:
            params["state"] = input_data.state
        if input_data.head:
            params["head"] = input_data.head
        if input_data.base:
            params["base"] = input_data.base
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
            prs_list = response.json()
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to list pull requests. Error: {e}")
            raise ToolExecutionException(
                f"Tool '{self.name}' failed to list pull requests for '{input_data.owner}/{input_data.repo}'. "
                f"Error: {str(e)}",
                recoverable=True,
            )

        if self.is_optimized_for_agents:
            lines = []
            for pr in prs_list:
                lines.append(f"PR #{pr.get('number')} - {pr.get('title')}")
            return {"content": f"Pull Requests for {input_data.owner}/{input_data.repo}:\n" + "\n".join(lines)}
        else:
            return {"content": prs_list}
