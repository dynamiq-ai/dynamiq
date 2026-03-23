from typing import ClassVar

from pydantic import BaseModel, Field

from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

from ..base import BaseAirtable


class ListRecordsInput(BaseModel):
    """
    Input schema for listing  records.
    Required:
      base_id, table_id_or_name
    Optional:
      view, pageSize, maxRecords, offset, etc.
    """

    base_id: str = Field(..., description="The ID of the  base.")
    table_id_or_name: str = Field(..., description="The table name or table ID.")
    view: str | None = Field(None, description="View name or ID to filter/sort records.")
    page_size: int | None = Field(None, description="Number of records per page (<= 100).")
    max_records: int | None = Field(None, description="Max total number of records to return.")
    offset: str | None = Field(None, description="Offset token from previous page.")
    filter_by_formula: str | None = Field(None, alias="filterByFormula", description=" formula for filtering records.")
    sort_field: str | None = Field(None, description="Field name or ID to sort on.")
    sort_direction: str | None = Field(None, description="Either 'asc' or 'desc' if sort_field is used.")

    class Config:
        allow_population_by_field_name = True


class ListRecords(BaseAirtable):
    """
    Node to list records from an  base/table.
    """

    name: str = "ListRecords"
    description: str = "Lists records from a specified base/table in ."
    input_schema: ClassVar[type[ListRecordsInput]] = ListRecordsInput

    def execute(self, input_data: ListRecordsInput, config: RunnableConfig = None, **kwargs):
        """
        Hit the GET https://api.airtable.com/v0/{base_id}/{table_id_or_name}
        with the appropriate query parameters, then return results.
        """
        logger.info(f"Node {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")
        self.run_on_node_execute_run(config.callbacks if config else [], **kwargs)

        base_id = input_data.base_id
        table = input_data.table_id_or_name

        # Build the URL
        api_url = f"{self.base_url}/{base_id}/{table}"

        # Build query params
        params = {}
        if input_data.view:
            params["view"] = input_data.view
        if input_data.page_size:
            params["pageSize"] = input_data.page_size
        if input_data.max_records:
            params["maxRecords"] = input_data.max_records
        if input_data.offset:
            params["offset"] = input_data.offset
        if input_data.filter_by_formula:
            params["filterByFormula"] = input_data.filter_by_formula
        if input_data.sort_field:
            params["sort[0][field]"] = input_data.sort_field
            if input_data.sort_direction in ("asc", "desc"):
                params["sort[0][direction]"] = input_data.sort_direction

        try:
            response = self.client.get(api_url, headers=self.connection.headers, params=params)
            response.raise_for_status()
            records_data = response.json()
        except Exception as e:
            logger.error(f"Node {self.name} - {self.id}: Failed to list records. Error: {e}")
            raise ToolExecutionException(
                f"Node {self.name} encountered an error while listing records: {e}",
                recoverable=True,
            )

        if self.is_optimized_for_agents:
            content = ""
            for record in records_data.get("records", []):
                content += f"ID: {record['id']}\n"
                for field, value in record["fields"].items():
                    content += f"{field}: {value}\n"
                content += "\n"
            return {"content": content}
        else:
            return {"content": records_data}
