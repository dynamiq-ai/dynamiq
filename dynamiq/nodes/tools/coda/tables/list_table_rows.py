from typing import ClassVar

from pydantic import BaseModel, Field

from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

from ..base import BaseCodaNode


class CodaListTableRowsInputSchema(BaseModel):
    """
    Input schema for listing rows in a Coda table.
    """

    doc_id: str = Field(..., alias="docId", description="ID of the doc.")
    table_id_or_name: str = Field(..., alias="tableIdOrName", description="ID or name of the table. (Name is fragile.)")
    query: str | None = Field(
        default=None,
        description=(
            "Query used to filter returned rows, specified as "
            "<column_id_or_name>:<value>. E.g.: "
            '"My Column":"Apple" or c-tuVwxYz:"Apple"'
        ),
    )
    sort_by: str | None = Field(
        default=None,
        alias="sortBy",
        description=("Sort order of the rows returned. One of: 'createdAt', 'updatedAt', " "or 'natural'."),
    )
    use_column_names: bool | None = Field(
        default=None, alias="useColumnNames", description="If true, uses column names rather than IDs in output."
    )
    value_format: str | None = Field(
        default=None,
        alias="valueFormat",
        description=("The format that cell values are returned as. One of 'simple', " "'simpleWithArrays', 'rich'."),
    )
    visible_only: bool | None = Field(
        default=None, alias="visibleOnly", description="If true, returns only visible rows and columns for the table."
    )
    limit: int | None = Field(default=25, description="Maximum number of results to return.")
    page_token: str | None = Field(
        default=None, alias="pageToken", description="Token to fetch the next page of results."
    )
    sync_token: str | None = Field(
        default=None, alias="syncToken", description="Token from a previous call, to fetch new or updated rows."
    )

    class Config:
        populate_by_name = True


class CodaListTableRows(BaseCodaNode):
    """
    Node to list rows in a specified table of a Coda doc.
    """

    name: str = "CodaListTableRows"
    description: str = "Fetch a list of rows from a given table in a Coda doc."
    input_schema: ClassVar[type[CodaListTableRowsInputSchema]] = CodaListTableRowsInputSchema

    def execute(self, input_data: CodaListTableRowsInputSchema, config: RunnableConfig | None = None, **kwargs):
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        doc_id = input_data.doc_id
        table_id_or_name = input_data.table_id_or_name
        api_url = f"{self.connection.url}docs/{doc_id}/tables/{table_id_or_name}/rows"

        params = self.dict_to_camel_case(input_data.model_dump(exclude_none=True, by_alias=True))

        try:
            response = self.client.get(api_url, headers=self.connection.headers, params=params)
            response.raise_for_status()
            rows_result = response.json()
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to list table rows. Error: {e}")
            raise ToolExecutionException(
                f"Tool '{self.name}' failed to list table rows. Error: {str(e)}.",
                recoverable=True,
            )

        if self.is_optimized_for_agents:
            items = rows_result.get("items", [])
            if not items:
                return {"content": "No rows found for this table."}

            lines = []
            for row in items:
                row_id = row.get("id")
                row_name = row.get("name")
                index = row.get("index")
                lines.append(f"- Row ID: {row_id}, Name: {row_name}, Index: {index}")

            content_str = "Rows:\n" + "\n".join(lines)
            return {"content": content_str}
        else:
            return {"content": rows_result}
