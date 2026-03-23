from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

from ..base import BaseGitHub


class ListAuthenticatedUserRepositoriesInputSchema(BaseModel):
    """
    Input parameters for listing repositories for the authenticated user.

    Docs: https://docs.github.com/en/rest/repos/repos#list-repositories-for-the-authenticated-user
    """

    visibility: Literal["all", "public", "private"] | None = Field(
        default=None, description="Filter by repo visibility: all, public, or private. Default is 'all'."
    )
    affiliation: str | None = Field(
        default=None,
        description=(
            "Comma-separated list of affiliations. Can be 'owner', 'collaborator', 'organization_member'. "
            "Default: 'owner,collaborator,organization_member'."
        ),
    )
    type: Literal["all", "owner", "public", "private", "member"] | None = Field(
        default=None,
        description=(
            "Filter by type. Will cause a 422 error if used with 'visibility' or 'affiliation'. " "Default: 'all'."
        ),
    )
    sort: Literal["created", "updated", "pushed", "full_name"] | None = Field(
        default="full_name", description="Field to sort by. Default is 'full_name'."
    )
    direction: Literal["asc", "desc"] | None = Field(
        default=None, description="Order to sort results. Default: 'asc' if sort=full_name, otherwise 'desc'."
    )
    per_page: int | None = Field(default=None, description="Number of results per page (max 100). Default is 30.")
    page: int | None = Field(default=None, description="Page number of results to fetch. Default is 1.")
    since: str | None = Field(default=None, description="Only show repos updated after this time (ISO8601).")
    before: str | None = Field(default=None, description="Only show repos updated before this time (ISO8601).")


class ListAuthenticatedUserRepositories(BaseGitHub):
    """
    Calls GET /user/repos to list repositories for the authenticated user.
    """

    name: str = "GitHubListAuthenticatedUserRepositories"
    description: str = (
        "Lists repositories that the authenticated user has explicit permission to access. "
        "This includes repos owned by the user, collaborator repos, and org repos."
    )
    input_schema: ClassVar[type[ListAuthenticatedUserRepositoriesInputSchema]] = (
        ListAuthenticatedUserRepositoriesInputSchema
    )

    def execute(
        self,
        input_data: ListAuthenticatedUserRepositoriesInputSchema,
        config: RunnableConfig | None = None,
        **kwargs,
    ):
        logger.info(f"Node {self.name} - {self.id}: started with input {input_data.dict()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        api_url = f"{self.connection.url}/user/repos"
        params = {}
        if input_data.visibility:
            params["visibility"] = input_data.visibility
        if input_data.affiliation:
            params["affiliation"] = input_data.affiliation
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
        if input_data.since:
            params["since"] = input_data.since
        if input_data.before:
            params["before"] = input_data.before

        try:
            response = self.client.get(api_url, headers=self.connection.headers, params=params)
            response.raise_for_status()
            repos_list = response.json()
        except Exception as e:
            logger.error(f"Node {self.name} - {self.id}: failed to list auth user repos. Error: {e}")
            raise ToolExecutionException(
                f"Tool '{self.name}' failed to list repositories for the authenticated user. Error: {str(e)}",
                recoverable=True,
            )

        if self.is_optimized_for_agents:
            lines = [f"- {repo.get('full_name')}" for repo in repos_list]
            return {"content": "Repositories for authenticated user:\n" + "\n".join(lines)}
        else:
            return {"content": repos_list}
