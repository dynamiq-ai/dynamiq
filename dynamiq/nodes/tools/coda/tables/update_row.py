from typing import ClassVar

from pydantic import BaseModel, Field

from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

from ..base import BaseCodaNode, CellEdit


class CodaUpdateRowInputSchema(BaseModel):
    """
    Input schema for updating a single row in a Coda table.
    """

    doc_id: str = Field(..., alias="docId", description="ID of the doc.")
    table_id_or_name: str = Field(..., alias="tableIdOrName", description="ID or name of the table. (Name is fragile.)")
    row_id_or_name: str = Field(
        ..., alias="rowIdOrName", description="ID or name of the row to update. (Name is fragile.)"
    )
    disable_parsing: bool | None = Field(
        default=None, alias="disableParsing", description="If true, the API will not attempt to parse the data."
    )
    cells: list[CellEdit] = Field(..., description="List of cell edits for the row.")

    class Config:
        populate_by_name = True


class CodaUpdateRow(BaseCodaNode):
    """
    Node to update an existing row in a Coda table.
    """

    name: str = "CodaUpdateRow"
    description: str = "Update a specific row in a Coda table."
    input_schema: ClassVar[type[CodaUpdateRowInputSchema]] = CodaUpdateRowInputSchema

    def execute(self, input_data: CodaUpdateRowInputSchema, config: RunnableConfig | None = None, **kwargs):
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        doc_id = input_data.doc_id
        table_id_or_name = input_data.table_id_or_name
        row_id_or_name = input_data.row_id_or_name
        api_url = f"{self.connection.url}docs/{doc_id}/tables/{table_id_or_name}/rows/{row_id_or_name}"

        # Query params
        query_params = {}
        if input_data.disable_parsing is not None:
            query_params["disableParsing"] = str(input_data.disable_parsing).lower()

        # Build the JSON body
        # The endpoint expects: { "row": { "cells": [ ... ] } }
        row_edit_payload = {"row": {"cells": [cell.model_dump(exclude_none=True) for cell in input_data.cells]}}

        try:
            response = self.client.put(
                api_url, headers=self.connection.headers, json=row_edit_payload, params=query_params
            )
            response.raise_for_status()
            result = response.json()
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to update row. Error: {e}")
            raise ToolExecutionException(
                f"Tool '{self.name}' failed to update the row. Error: {str(e)}.",
                recoverable=True,
            )

        if self.is_optimized_for_agents:
            return {
                "content": (
                    f"Row updated successfully.\n"
                    f"Request ID: {result.get('requestId')}\n"
                    f"Row ID: {result.get('id')}\n"
                )
            }
        else:
            return {"content": result}
