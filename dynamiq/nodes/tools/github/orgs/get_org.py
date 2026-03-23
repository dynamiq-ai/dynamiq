from typing import ClassVar

from pydantic import BaseModel, Field

from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

from ..base import BaseGitHub


class GetOrganizationInputSchema(BaseModel):
    """
    Input parameters for getting a single organization by name.
    """

    org: str = Field(..., description="The organization name (not case-sensitive).")


class GetOrganization(BaseGitHub):
    """
    Calls GET /orgs/{org} to get details of a single GitHub organization.
    """

    name: str = "GitHubGetOrganization"
    description: str = "Get information about a specific GitHub organization."
    input_schema: ClassVar[type[GetOrganizationInputSchema]] = GetOrganizationInputSchema

    def execute(self, input_data: GetOrganizationInputSchema, config: RunnableConfig | None = None, **kwargs):
        logger.info(f"Tool {self.name} - {self.id}: started with input: {input_data.dict()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        api_url = f"{self.connection.url}/orgs/{input_data.org}"

        try:
            response = self.client.get(api_url, headers=self.connection.headers)
            response.raise_for_status()
            org_details = response.json()
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to get organization. Error: {e}")
            raise ToolExecutionException(
                f"Tool '{self.name}' failed to get organization {input_data.org}. Error: {str(e)}",
                recoverable=True,
            )

        if self.is_optimized_for_agents:
            summary_text = (
                f"Organization: {org_details.get('login', '')}\n"
                f"ID: {org_details.get('id', '')}\n"
                f"Public Repos: {org_details.get('public_repos', '')}\n"
                f"Description: {org_details.get('description', '')}\n"
                f"Blog: {org_details.get('blog', '')}\n"
                f"Location: {org_details.get('location', '')}\n"
                f"Email: {org_details.get('email', '')}\n"
                f"Type: {org_details.get('type', '')}\n"
            )
            return {"content": summary_text}

        return {"content": org_details}
