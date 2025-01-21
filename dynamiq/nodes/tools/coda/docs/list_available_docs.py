from typing import ClassVar

from pydantic import BaseModel, Field

from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

from ..base import BaseCodaNode


class CodaListAvailableDocsInputSchema(BaseModel):
    """
    Input schema for the CodaListAvailableDocs node.
    """

    is_owner: bool | None = Field(default=None, description="Show only docs owned by the user.")
    is_published: bool | None = Field(default=None, description="Show only published docs.")
    query: str | None = Field(default=None, description="Search term to filter docs.")
    source_doc: str | None = Field(default=None, description="Show only docs copied from the specified doc ID.")
    is_starred: bool | None = Field(default=None, description="Return starred docs if True.")
    in_gallery: bool | None = Field(default=None, description="Show only docs in the gallery.")
    workspace_id: str | None = Field(default=None, description="Show only docs in the given workspace.")
    folder_id: str | None = Field(default=None, description="Show only docs in the given folder.")
    limit: int | None = Field(default=25, description="Maximum number of results to return.")
    page_token: str | None = Field(default=None, description="Token to fetch the next page of results.")


class CodaListAvailableDocs(BaseCodaNode):
    """
    Node to list available Coda docs using the Coda API.
    """

    name: str = "CodaListAvailableDocs"
    description: str = "Fetch a list of available Coda docs."
    input_schema: ClassVar[type[CodaListAvailableDocsInputSchema]] = CodaListAvailableDocsInputSchema

    def execute(self, input_data: CodaListAvailableDocsInputSchema, config: RunnableConfig | None = None, **kwargs):
        """
        Execute the node to fetch Coda docs.

        Args:
            input_data (CodaListAvailableDocsInputSchema): Input schema data.
            config (RunnableConfig, optional): Execution configuration.

        Returns:
            dict: Result of the API call.
        """
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        api_url = self.connection.url + "docs"
        params = self.dict_to_camel_case(input_data.model_dump(exclude_none=True))

        try:
            response = self.client.get(api_url, headers=self.connection.headers, params=params)
            response.raise_for_status()
            docs_result = response.json()
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to fetch docs. Error: {e}")
            raise ToolExecutionException(
                f"Tool '{self.name}' failed to execute the requested action. Error: {str(e)}. "
                f"Please analyze the error and take appropriate action.",
                recoverable=True,
            )

        if self.is_optimized_for_agents:
            docs_list = docs_result.get("items", [])
            optimized_result = "\n".join(f"- Doc Name: {doc['name']} (ID: {doc['id']})" for doc in docs_list)
            return {"content": optimized_result}
        else:
            result = {
                "docs": docs_result.get("items", []),
                "next_page_token": docs_result.get("nextPageToken"),
            }
            return {"content": result}
