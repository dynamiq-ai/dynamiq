from typing import ClassVar

from pydantic import BaseModel, Field

from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

from ..base import BaseCodaNode


class CodaGetRowInputSchema(BaseModel):
    """
    Input schema for fetching details about a single row in a Coda table.
    """

    doc_id: str = Field(..., alias="docId", description="ID of the doc.")
    table_id_or_name: str = Field(..., alias="tableIdOrName", description="ID or name of the table. (Name is fragile.)")
    row_id_or_name: str = Field(
        ...,
        alias="rowIdOrName",
        description=(
            "ID or name of the row. (Name is fragile.) If multiple rows have this name, "
            "an arbitrary matching row is returned."
        ),
    )
    use_column_names: bool | None = Field(
        default=None,
        alias="useColumnNames",
        description="If true, column names are used instead of IDs in the returned values.",
    )
    value_format: str | None = Field(
        default=None,
        alias="valueFormat",
        description=("The format that cell values are returned as. One of 'simple', " "'simpleWithArrays', 'rich'."),
    )

    class Config:
        populate_by_name = True


class CodaGetTableRows(BaseCodaNode):
    """
    Node to retrieve details about a specific row in a Coda table.
    """

    name: str = "CodaGetRow"
    description: str = "Retrieve details about a specific row in a Coda table."
    input_schema: ClassVar[type[CodaGetRowInputSchema]] = CodaGetRowInputSchema

    def execute(self, input_data: CodaGetRowInputSchema, config: RunnableConfig | None = None, **kwargs):
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        doc_id = input_data.doc_id
        table_id_or_name = input_data.table_id_or_name
        row_id_or_name = input_data.row_id_or_name

        api_url = f"{self.connection.url}docs/{doc_id}/tables/{table_id_or_name}/rows/{row_id_or_name}"

        params = self.dict_to_camel_case(input_data.model_dump(exclude_none=True, by_alias=True))

        try:
            response = self.client.get(api_url, headers=self.connection.headers, params=params)
            response.raise_for_status()
            row_info = response.json()
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to get row. Error: {e}")
            raise ToolExecutionException(
                f"Tool '{self.name}' failed to retrieve row details. Error: {str(e)}.",
                recoverable=True,
            )

        if self.is_optimized_for_agents:
            return {
                "content": (
                    f"Row Info:\n"
                    f"ID: {row_info.get('id')}\n"
                    f"Name: {row_info.get('name')}\n"
                    f"Index: {row_info.get('index')}\n"
                    f"Created At: {row_info.get('createdAt')}\n"
                    f"Updated At: {row_info.get('updatedAt')}\n"
                )
            }
        else:
            return {"content": row_info}
