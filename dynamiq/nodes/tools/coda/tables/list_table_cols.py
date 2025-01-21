from typing import ClassVar

from pydantic import BaseModel, Field

from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

from ..base import BaseCodaNode


class CodaListColumnsInputSchema(BaseModel):
    """
    Input schema for listing columns in a Coda table.
    """

    doc_id: str = Field(..., alias="docId", description="ID of the doc.")
    table_id_or_name: str = Field(..., alias="tableIdOrName", description="ID or name of the table. (Name is fragile.)")
    limit: int | None = Field(default=25, description="Maximum number of results to return (1-100).")
    page_token: str | None = Field(
        default=None, alias="pageToken", description="Token to fetch the next page of results."
    )
    visible_only: bool | None = Field(
        default=None, alias="visibleOnly", description="If true, returns only visible columns for the table."
    )

    class Config:
        populate_by_name = True


class CodaListTableColumns(BaseCodaNode):
    """
    Node to list columns in a specific table of a Coda doc.
    """

    name: str = "CodaListColumns"
    description: str = "Fetch a list of columns from a table in a Coda doc."
    input_schema: ClassVar[type[CodaListColumnsInputSchema]] = CodaListColumnsInputSchema

    def execute(self, input_data: CodaListColumnsInputSchema, config: RunnableConfig | None = None, **kwargs):
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        doc_id = input_data.doc_id
        table_id_or_name = input_data.table_id_or_name
        api_url = f"{self.connection.url}docs/{doc_id}/tables/{table_id_or_name}/columns"

        params = self.dict_to_camel_case(input_data.model_dump(exclude_none=True, by_alias=True))

        try:
            response = self.client.get(api_url, headers=self.connection.headers, params=params)
            response.raise_for_status()
            columns_result = response.json()
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to list columns. Error: {e}")
            raise ToolExecutionException(
                f"Tool '{self.name}' failed to list columns. Error: {str(e)}.",
                recoverable=True,
            )

        if self.is_optimized_for_agents:
            items = columns_result.get("items", [])
            if not items:
                return {"content": "No columns found for this table."}

            # Build a simple summary
            lines = []
            for col in items:
                col_id = col.get("id")
                col_name = col.get("name")
                col_type = col.get("format", {}).get("type", "unknown")
                lines.append(f"- Column ID: {col_id}, Name: {col_name}, Format Type: {col_type}")

            content_str = "Columns:\n" + "\n".join(lines)
            return {"content": content_str}
        else:
            return {"content": columns_result}
