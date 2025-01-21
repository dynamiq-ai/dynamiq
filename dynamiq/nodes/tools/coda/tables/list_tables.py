from typing import ClassVar

from pydantic import BaseModel, Field

from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

from ..base import BaseCodaNode


class CodaListTablesInputSchema(BaseModel):
    """
    Input schema for listing tables in a Coda doc.
    """

    doc_id: str = Field(..., alias="docId", description="ID of the doc.")
    limit: int | None = Field(default=25, description="Maximum number of tables/views to return.")
    page_token: str | None = Field(
        default=None, alias="pageToken", description="Token to fetch the next page of results."
    )
    sort_by: str | None = Field(
        default=None, alias="sortBy", description="Determines how to sort the tables (e.g., 'name')."
    )
    table_types: str | None = Field(
        default=None, alias="tableTypes", description="Comma-separated list of table types: 'table', 'view'."
    )

    class Config:
        populate_by_name = True


class CodaListTables(BaseCodaNode):
    """
    Node to list tables or views in a Coda doc.
    """

    name: str = "CodaListTables"
    description: str = "Fetch a list of tables/views from a given Coda doc."
    input_schema: ClassVar[type[CodaListTablesInputSchema]] = CodaListTablesInputSchema

    def execute(self, input_data: CodaListTablesInputSchema, config: RunnableConfig | None = None, **kwargs):
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        doc_id = input_data.doc_id
        api_url = f"{self.connection.url}docs/{doc_id}/tables"

        params = self.dict_to_camel_case(input_data.model_dump(exclude_none=True, by_alias=True))

        try:
            response = self.client.get(api_url, headers=self.connection.headers, params=params)
            response.raise_for_status()
            tables_result = response.json()
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to list tables. Error: {e}")
            raise ToolExecutionException(
                f"Tool '{self.name}' failed to list tables. Error: {str(e)}.",
                recoverable=True,
            )

        if self.is_optimized_for_agents:
            tables_list = tables_result.get("items", [])
            if not tables_list:
                return {"content": "No tables or views found in this doc."}

            optimized_result = "Tables / Views:\n" + "\n".join(
                f"- {tbl.get('name')} (ID: {tbl.get('id')}, Type: {tbl.get('tableType')})" for tbl in tables_list
            )
            return {"content": optimized_result}
        else:
            return {"content": tables_result}
