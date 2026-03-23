from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.tools.google_sheets.google_sheets_base import GoogleSheetsBase
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

DESCRIPTION_LOOKUP_SPREADSHEET_ROW = """## Lookup a Row in a Spreadsheet
Return the first row in a Google Sheets spreadsheet whose cells contain `query` (exact match in any column).

### Parameters
- `spreadsheet_id` (str): ID of the spreadsheet to lookup the row in.
- `query` (str): Value to query.
- `range` (str): Range of the spreadsheet to lookup the row in.
- `case_sensitive` (bool, optional): Whether to match the query case-sensitively. Defaults to False.
"""


class LookupSpreadsheetRowInputSchema(BaseModel):
    spreadsheet_id: str = Field(..., description="ID of the spreadsheet to get values from.")
    query: str = Field(..., description="Value to query.")
    range: str | None = Field(
        "A1:ZZ",
        description="Range of the spreadsheet to lookup the row in. Defaults to the entire spreadsheet.",
    )
    case_sensitive: bool | None = Field(False, description="Whether to match the query case-sensitively.")

    model_config = {"arbitrary_types_allowed": True}


class LookupSpreadsheetRow(GoogleSheetsBase):
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "lookup-spreadsheet-row"
    description: str = DESCRIPTION_LOOKUP_SPREADSHEET_ROW
    input_schema: ClassVar[type[BaseModel]] = LookupSpreadsheetRowInputSchema

    def execute(
        self,
        input_data: LookupSpreadsheetRowInputSchema,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Return the first row in a Google Sheets spreadsheet whose cells contain `query` (exact match in any column).

        Args:
            input_data (BatchGetValuesInputSchema): Spreadsheet details.
            config (RunnableConfig | None): Optional runtime config.
            **kwargs: Additional arguments.

        Returns:
            dict[str, Any]: The row with the given query.

        Raises:
            ToolExecutionException: If values retrieval fails.
        """
        logger.info(
            f"{self.name} ({self.id}) - Looking up row in range {input_data.range} "
            f"of spreadsheet {input_data.spreadsheet_id} with query {input_data.query}"
        )
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        try:
            rows = (
                self.client.spreadsheets()
                .values()
                .batchGet(
                    spreadsheetId=input_data.spreadsheet_id,
                    ranges=input_data.range,
                )
                .execute()
            )

            values = rows.get("valueRanges", [])[0].get("values", [])

            result = None

            for idx, row in enumerate(values, start=1):
                match = (
                    (lambda x: x == input_data.query)
                    if input_data.case_sensitive
                    else (lambda x: str(x).lower() == input_data.query.lower())
                )

                if any(match(cell) for cell in row):
                    logger.info(f"Found row {idx} with values {row}")

                    result = {"row_index": idx, "values": row}
                    break

            logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")
            return {"content": result}

        except Exception as e:
            raise ToolExecutionException(str(e), recoverable=True)
