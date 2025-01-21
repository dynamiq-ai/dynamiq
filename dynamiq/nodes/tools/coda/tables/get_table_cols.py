from typing import ClassVar

from pydantic import BaseModel, Field

from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

from ..base import BaseCodaNode


class CodaGetColumnInputSchema(BaseModel):
    """
    Input schema for fetching details about a single column in a Coda table.
    """

    doc_id: str = Field(..., alias="docId", description="ID of the doc.")
    table_id_or_name: str = Field(..., alias="tableIdOrName", description="ID or name of the table. (Name is fragile.)")
    column_id_or_name: str = Field(
        ..., alias="columnIdOrName", description="ID or name of the column. (Name is fragile.)"
    )

    class Config:
        populate_by_name = True


class CodaGetTableColumn(BaseCodaNode):
    """
    Node to retrieve details about a single column in a Coda table.
    """

    name: str = "CodaGetColumn"
    description: str = "Get details about a specific column in a Coda table."
    input_schema: ClassVar[type[CodaGetColumnInputSchema]] = CodaGetColumnInputSchema

    def execute(self, input_data: CodaGetColumnInputSchema, config: RunnableConfig | None = None, **kwargs):
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        doc_id = input_data.doc_id
        table_id_or_name = input_data.table_id_or_name
        column_id_or_name = input_data.column_id_or_name

        api_url = f"{self.connection.url}docs/{doc_id}/tables/{table_id_or_name}/columns/{column_id_or_name}"

        try:
            response = self.client.get(api_url, headers=self.connection.headers)
            response.raise_for_status()
            column_info = response.json()
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to get column. Error: {e}")
            raise ToolExecutionException(
                f"Tool '{self.name}' failed to get column details. Error: {str(e)}.",
                recoverable=True,
            )

        if self.is_optimized_for_agents:
            col_id = column_info.get("id")
            col_name = column_info.get("name")
            col_type = (column_info.get("format") or {}).get("type", "unknown")
            col_formula = column_info.get("formula")

            content_str = (
                f"Column Info:\n"
                f"ID: {col_id}\n"
                f"Name: {col_name}\n"
                f"Format Type: {col_type}\n"
                f"Formula: {col_formula}\n"
            )
            return {"content": content_str}
        else:
            return {"content": column_info}
