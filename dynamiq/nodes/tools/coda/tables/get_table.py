from typing import ClassVar

from pydantic import BaseModel, Field

from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

from ..base import BaseCodaNode


class CodaGetTableInputSchema(BaseModel):
    """
    Input schema for fetching details about a specific table or view.
    """

    doc_id: str = Field(..., alias="docId", description="ID of the doc.")
    table_id_or_name: str = Field(..., alias="tableIdOrName", description="ID or name of the table/view.")
    use_updated_table_layouts: bool | None = Field(
        default=None,
        alias="useUpdatedTableLayouts",
        description="If true, returns 'detail' and 'form' for layout field of detail/form layouts.",
    )

    class Config:
        populate_by_name = True


class CodaGetTable(BaseCodaNode):
    """
    Node to get details about a specific table or view in a Coda doc.
    """

    name: str = "CodaGetTable"
    description: str = "Fetch detailed info about a table/view in a Coda doc."
    input_schema: ClassVar[type[CodaGetTableInputSchema]] = CodaGetTableInputSchema

    def execute(self, input_data: CodaGetTableInputSchema, config: RunnableConfig | None = None, **kwargs):
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        doc_id = input_data.doc_id
        table_id_or_name = input_data.table_id_or_name
        api_url = f"{self.connection.url}docs/{doc_id}/tables/{table_id_or_name}"

        params = self.dict_to_camel_case(input_data.model_dump(exclude_none=True, by_alias=True))

        try:
            response = self.client.get(api_url, headers=self.connection.headers, params=params)
            response.raise_for_status()
            table_info = response.json()
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to get table. Error: {e}")
            raise ToolExecutionException(
                f"Tool '{self.name}' failed to get table details. Error: {str(e)}.",
                recoverable=True,
            )

        if self.is_optimized_for_agents:
            return {
                "content": (
                    f"Table Info:\n"
                    f"Name: {table_info.get('name')}\n"
                    f"ID: {table_info.get('id')}\n"
                    f"Type: {table_info.get('tableType')}\n"
                    f"Row Count: {table_info.get('rowCount')}\n"
                    f"Layout: {table_info.get('layout')}\n"
                )
            }
        else:
            return {"content": table_info}
