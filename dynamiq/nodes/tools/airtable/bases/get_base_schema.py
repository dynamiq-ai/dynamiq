from typing import ClassVar

from pydantic import BaseModel, Field

from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

from ..base import BaseAirtable


class GetBaseSchemaInput(BaseModel):
    """
    Input schema for retrieving the schema of a base (its tables, fields, etc.).
    GET /v0/meta/bases/{baseId}/tables
    """

    base_id: str = Field(..., description="The ID of the  base.")
    include_visible_field_ids: bool = Field(
        default=False, description="If true, adds 'include=visibleFieldIds' param (views of type grid only)."
    )


class GetBaseSchema(BaseAirtable):
    """
    Node to get the schema of a base (tables, fields, views, etc.).
    """

    name: str = "GetBaseSchema"
    description: str = "Retrieves the schema of a specified base."
    input_schema: ClassVar[type[GetBaseSchemaInput]] = GetBaseSchemaInput

    def execute(self, input_data: GetBaseSchemaInput, config: RunnableConfig = None, **kwargs):
        logger.info(f"Node {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")
        callbacks = config.callbacks if config else []
        self.run_on_node_execute_run(callbacks, **kwargs)

        base_id = input_data.base_id
        url = f"{self.base_url}/meta/bases/{base_id}/tables"

        params = {}
        if input_data.include_visible_field_ids:
            params["include"] = "visibleFieldIds"

        try:
            response = self.client.get(url, headers=self.connection.headers, params=params)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"{self.name} - {self.id}: failed to get base schema. Error: {e}")
            raise ToolExecutionException(f"Failed to get base schema: {str(e)}", recoverable=True)

        if self.is_optimized_for_agents:
            tables = data.get("tables", [])
            content = ""

            for table in tables:
                table_name = table.get("name", "<unknown>")
                table_id = table.get("id", "<unknown>")
                table_desc = table.get("description", "")
                primary_field_id = table.get("primaryFieldId", "")

                content += f"\nTable: {table_name}"
                content += f"\n  ID: {table_id}"
                if table_desc:
                    content += f"\n  Description: {table_desc}"
                if primary_field_id:
                    content += f"\n  Primary Field ID: {primary_field_id}"

                fields = table.get("fields", [])
                if fields:
                    content += "\n  Fields:"
                    for field in fields:
                        field_name = field.get("name", "<unknown>")
                        field_id = field.get("id", "<unknown>")
                        field_type = field.get("type", "<unknown>")
                        field_desc = field.get("description", "")
                        content += f"\n    - Field: {field_name}"
                        content += f" (ID: {field_id}, Type: {field_type})"
                        if field_desc:
                            content += f"\n      Description: {field_desc}"

                views = table.get("views", [])
                if views:
                    content += "\n  Views:"
                    for view in views:
                        view_name = view.get("name", "<unknown>")
                        view_id = view.get("id", "<unknown>")
                        view_type = view.get("type", "<unknown>")
                        content += f"\n    - View: {view_name}"
                        content += f" (ID: {view_id}, Type: {view_type})"

                content += "\n" + ("-" * 40)
            return {"content": content}

        else:
            return {"content": data}
