from typing import ClassVar

from pydantic import BaseModel, Field

from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

from ..base import BaseCodaNode


class CodaListPagesInputSchema(BaseModel):
    """
    Input schema for listing pages in a Coda doc.
    """

    doc_id: str = Field(..., alias="docId", description="ID of the doc.")
    limit: int | None = Field(default=25, description="Maximum number of pages to return.")
    page_token: str | None = Field(
        default=None, alias="pageToken", description="Token to fetch the next page of results."
    )

    class Config:
        populate_by_name = True


class CodaListPages(BaseCodaNode):
    """
    Node to list pages in a Coda doc.
    """

    name: str = "CodaListPages"
    description: str = "Fetch a list of pages from a given Coda doc."
    input_schema: ClassVar[type[CodaListPagesInputSchema]] = CodaListPagesInputSchema

    def execute(self, input_data: CodaListPagesInputSchema, config: RunnableConfig | None = None, **kwargs):
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        # Prepare request
        doc_id = input_data.doc_id
        api_url = f"{self.connection.url}docs/{doc_id}/pages"
        params = self.dict_to_camel_case(input_data.model_dump(exclude_none=True, by_alias=True))

        try:
            response = self.client.get(api_url, headers=self.connection.headers, params=params)
            response.raise_for_status()
            pages_result = response.json()
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to list pages. Error: {e}")
            raise ToolExecutionException(
                f"Tool '{self.name}' failed to list pages. Error: {str(e)}.",
                recoverable=True,
            )

        if self.is_optimized_for_agents:
            pages_list = pages_result.get("items", [])
            if not pages_list:
                return {"content": "No pages found in this doc."}

            optimized_result = "Pages:\n" + "\n".join(
                f"- {page.get('name')} (ID: {page.get('id')})" for page in pages_list
            )
            return {"content": optimized_result}
        else:
            return {"content": pages_result}
