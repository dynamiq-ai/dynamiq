from typing import ClassVar

from pydantic import BaseModel, Field

from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

from ..base import BaseCodaNode, RowEdit


class CodaInsertUpsertRowsInputSchema(BaseModel):
    """
    Input schema for inserting/upserting rows in a Coda table.
    """

    doc_id: str = Field(..., alias="docId", description="ID of the doc.")
    table_id_or_name: str = Field(
        ..., alias="tableIdOrName", description="ID or name of the base table. (Name is fragile.)"
    )
    disable_parsing: bool | None = Field(
        default=None, alias="disableParsing", description="If true, the API will not attempt to parse the data."
    )
    rows: list[RowEdit] = Field(..., description="List of rows to insert/upsert.")
    key_columns: list[str] | None = Field(
        default=None,
        alias="keyColumns",
        description=(
            "Optional list of column identifiers used as upsert keys. If omitted, rows "
            "will be inserted. If provided, matching rows will be updated, otherwise inserted."
        ),
    )

    class Config:
        populate_by_name = True


class CodaInsertUpsertRows(BaseCodaNode):
    """
    Node to insert or upsert rows in a Coda table.
    """

    name: str = "CodaInsertUpsertRows"
    description: str = "Insert or upsert rows into a Coda table."
    input_schema: ClassVar[type[CodaInsertUpsertRowsInputSchema]] = CodaInsertUpsertRowsInputSchema

    def execute(self, input_data: CodaInsertUpsertRowsInputSchema, config: RunnableConfig | None = None, **kwargs):
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        doc_id = input_data.doc_id
        table_id_or_name = input_data.table_id_or_name
        api_url = f"{self.connection.url}docs/{doc_id}/tables/{table_id_or_name}/rows"

        # Query params
        query_params = {}
        if input_data.disable_parsing is not None:
            query_params["disableParsing"] = str(input_data.disable_parsing).lower()

        payload_dict = input_data.model_dump(exclude_none=True, by_alias=True)
        for remove_key in ("docId", "tableIdOrName", "disableParsing"):
            payload_dict.pop(remove_key, None)

        payload = payload_dict

        try:
            response = self.client.post(api_url, headers=self.connection.headers, json=payload, params=query_params)
            response.raise_for_status()
            result = response.json()
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to insert/upsert rows. Error: {e}")
            raise ToolExecutionException(
                f"Tool '{self.name}' failed to insert/upsert rows. Error: {str(e)}.",
                recoverable=True,
            )

        if self.is_optimized_for_agents:
            return {"content": (f"Rows Inserted/Upserted successfully.\n" f"Request ID: {result.get('requestId')}\n")}
        else:
            return {"content": result}
