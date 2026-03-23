from typing import Any, ClassVar

from pydantic import BaseModel, Field

from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

from ..base import BaseAirtable


class CreateRecordFields(BaseModel):
    """
    Represents the fields for a single record creation.
    You can expand on the types or keep them as `Any`.
    """

    # Example arbitrary fields. Replace with your actual field usage, or keep it flexible:
    fields: dict[str, Any] = Field(..., description="Dictionary of fieldName -> fieldValue.")


class CreateRecordInput(BaseModel):
    """
    Input schema for creating new  records.
    """

    base_id: str = Field(..., description="The ID of the  base.")
    table_id_or_name: str = Field(..., description="The table name or table ID.")
    # For multiple records:
    records: list[CreateRecordFields] | None = Field(
        default=None,
        description="A list of record field dictionaries to create multiple records.",
    )
    # Or a single record at top level:
    fields: dict[str, Any] | None = Field(
        default=None, description="If creating a single record, specify fields directly."
    )
    typecast: bool = Field(default=False, description="Enable  best-effort automatic data conversion for strings.")
    return_fields_by_field_id: bool = Field(
        default=False,
        alias="returnFieldsByFieldId",
        description="If true, fields in response keyed by field ID instead of field name.",
    )

    class Config:
        allow_population_by_field_name = True


class CreateRecord(BaseAirtable):
    """
    Node that creates a single or multiple records in .
    """

    name: str = "CreateRecord"
    description: str = "Create one or many records in ."
    input_schema: ClassVar[type[CreateRecordInput]] = CreateRecordInput

    def execute(self, input_data: CreateRecordInput, config: RunnableConfig = None, **kwargs):
        logger.info(f"Node {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")
        self.run_on_node_execute_run(config.callbacks if config else [], **kwargs)

        base_id = input_data.base_id
        table = input_data.table_id_or_name
        api_url = f"{self.base_url}/{base_id}/{table}"

        # Construct the payload for
        payload: dict[str, Any] = {
            "returnFieldsByFieldId": input_data.return_fields_by_field_id,
            "typecast": input_data.typecast,
        }

        if input_data.records:
            # Creating multiple records
            # 'records' is a list of dicts: each item is {"fields": {...}}
            # We can pass it directly:
            payload["records"] = [rec.model_dump() for rec in input_data.records]
        elif input_data.fields:
            # Creating a single record
            payload["fields"] = input_data.fields
        else:
            raise ValueError("Either 'records' or 'fields' must be provided to create a record.")

        try:
            response = self.client.post(
                api_url,
                headers=self.connection.headers,
                json=payload,
            )
            response.raise_for_status()
            create_data = response.json()
        except Exception as e:
            logger.error(f"Node {self.name} - {self.id}: failed to create record(s). Error: {e}")
            raise ToolExecutionException(
                f"Node {self.name} encountered an error while creating record(s): {e}",
                recoverable=True,
            )

        if self.is_optimized_for_agents:
            content = "Created record(s) with IDs:\n"
            if "records" in create_data and isinstance(create_data["records"], list):
                content += "\n".join([str(rec) for rec in create_data["records"]])
            else:
                content += "\n".join([str(create_data)])
            return {"content": content}
        else:
            return {"content": create_data}
