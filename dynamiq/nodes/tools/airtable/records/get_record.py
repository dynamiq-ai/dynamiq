from typing import ClassVar

from pydantic import BaseModel, Field

from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

from ..base import BaseAirtable


class GetRecordInput(BaseModel):
    """
    Input schema for retrieving a single  record.
    """

    base_id: str = Field(..., description="The ID of the  base.")
    table_id_or_name: str = Field(..., description="The table name or table ID.")
    record_id: str = Field(..., description="The record ID to retrieve.")

    cell_format: str | None = Field(
        None, alias="cellFormat", description="Allowed: 'json' or 'string'. Defaults to 'json'."
    )
    return_fields_by_field_id: bool | None = Field(
        None,
        alias="returnFieldsByFieldId",
        description="If true, fields object is keyed by field ID instead of field name.",
    )

    class Config:
        allow_population_by_field_name = True


class GetRecord(BaseAirtable):
    """
    Node to get a single record from .
    """

    name: str = "GetRecord"
    description: str = "Retrieves a single record by ID from ."
    input_schema: ClassVar[type[GetRecordInput]] = GetRecordInput

    def execute(self, input_data: GetRecordInput, config: RunnableConfig = None, **kwargs):
        logger.info(f"Node {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")
        self.run_on_node_execute_run(config.callbacks if config else [], **kwargs)

        base_id = input_data.base_id
        table = input_data.table_id_or_name
        record_id = input_data.record_id

        api_url = f"{self.base_url}/{base_id}/{table}/{record_id}"

        params = {}
        if input_data.cell_format:
            params["cellFormat"] = input_data.cell_format
        if input_data.return_fields_by_field_id is not None:
            params["returnFieldsByFieldId"] = input_data.return_fields_by_field_id

        try:
            response = self.client.get(api_url, headers=self.connection.headers, params=params)
            response.raise_for_status()
            record_data = response.json()
        except Exception as e:
            logger.error(f"Node {self.name} - {self.id}: Failed to retrieve record. Error: {e}")
            raise ToolExecutionException(
                f"Node {self.name} encountered an error while retrieving record: {e}",
                recoverable=True,
            )

        if self.is_optimized_for_agents:
            return {
                "content": (
                    f"Record ID: {record_data['id']}\n"
                    f"CreatedTime: {record_data.get('createdTime')}\n"
                    f"Fields: {record_data.get('fields', {})}"
                )
            }
        else:
            return {"content": record_data}
