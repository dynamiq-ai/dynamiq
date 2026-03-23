from typing import ClassVar

from pydantic import BaseModel, Field

from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

from ..base import BaseGitHub


class PullRequestCommitsInputSchema(BaseModel):
    """
    Input parameters for listing commits in a specific pull request.
    """

    owner: str = Field(..., description="The account owner of the repository.")
    repo: str = Field(..., description="The name of the repository (not case sensitive).")
    pull_number: int = Field(..., description="The pull request number.")
    per_page: int | None = Field(default=None, description="Number of results per page (max 100). Default is 30.")
    page: int | None = Field(default=None, description="Page number of the results to fetch. Default is 1.")


class ListPullRequestCommits(BaseGitHub):
    """
    Calls GET /repos/{owner}/{repo}/pulls/{pull_number}/commits
    to list commits on a pull request.
    """

    name: str = "GitHubListPullRequestCommits"
    description: str = "List commits on a specific pull request."
    input_schema: ClassVar[type[PullRequestCommitsInputSchema]] = PullRequestCommitsInputSchema

    def execute(self, input_data: PullRequestCommitsInputSchema, config: RunnableConfig | None = None, **kwargs):
        logger.info(f"Tool {self.name} - {self.id}: started with input: {input_data.dict()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        api_url = (
            f"{self.connection.url}/repos/"
            f"{input_data.owner}/{input_data.repo}/pulls/{input_data.pull_number}/commits"
        )

        params = {}
        if input_data.per_page:
            params["per_page"] = input_data.per_page
        if input_data.page:
            params["page"] = input_data.page

        try:
            response = self.client.get(api_url, headers=self.connection.headers, params=params)
            response.raise_for_status()
            commits_list = response.json()
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to list PR commits. Error: {e}")
            raise ToolExecutionException(
                f"Tool '{self.name}' failed to list commits for PR #{input_data.pull_number} "
                f"in repo '{input_data.owner}/{input_data.repo}'. Error: {str(e)}",
                recoverable=True,
            )

        if self.is_optimized_for_agents:
            lines = []
            for commit in commits_list:
                sha = commit.get("sha", "")
                msg = commit.get("commit", {}).get("message", "")
                lines.append(f"- {sha[:7]}: {msg}")
            return {
                "content": (
                    f"Commits for PR #{input_data.pull_number} in {input_data.owner}/{input_data.repo}:\n"
                    + "\n".join(lines)
                )
            }
        else:
            return {"content": commits_list}
