from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.tools.google_sheets.google_sheets_base import GoogleSheetsBase
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

DESCRIPTION_GET_SHEET_NAMES = """## Get Sheet Names in a Spreadsheet
Get the names of the sheets in a Google Sheets spreadsheet.

### Parameters
- `spreadsheet_id` (str): ID of the spreadsheet to get the sheet names from.
"""


class GetSheetNamesInputSchema(BaseModel):
    spreadsheet_id: str = Field(..., description="ID of the spreadsheet to get the sheet names from.")

    model_config = {"arbitrary_types_allowed": True}


class GetSheetNames(GoogleSheetsBase):
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "get-sheet-names"
    description: str = DESCRIPTION_GET_SHEET_NAMES
    input_schema: ClassVar[type[BaseModel]] = GetSheetNamesInputSchema

    def execute(
        self,
        input_data: GetSheetNamesInputSchema,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Retrieves the names of the sheets in a Google Sheets spreadsheet.

        Args:
            input_data (CreateSpreadsheetInputSchema): Spreadsheet details.
            config (RunnableConfig | None): Optional runtime config.
            **kwargs: Additional arguments.

        Returns:
            dict[str, Any]: The names of the sheets in the spreadsheet.

        Raises:
            ToolExecutionException: If sheet names retrieval fails.
        """
        logger.info(f"{self.name} ({self.id}) - Getting sheet names from spreadsheet {input_data.spreadsheet_id}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        try:
            result = (
                self.client.spreadsheets()
                .get(
                    spreadsheetId=input_data.spreadsheet_id,
                    fields="sheets.properties.title",
                )
                .execute()
            )

            sheet_names = [sheet.get("properties", {}).get("title") for sheet in result.get("sheets", [])]
            logger.info(f"Sheet names retrieved from spreadsheet {input_data.spreadsheet_id}: {sheet_names}")

            logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")
            return {"content": sheet_names}

        except Exception as e:
            raise ToolExecutionException(str(e), recoverable=True)
