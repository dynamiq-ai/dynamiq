from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.tools.google_sheets.google_sheets_base import GoogleSheetsBase
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

DESCRIPTION_GET_SPREADSHEET_INFO = """## Get Spreadsheet Info
Get the comprehensive metadata of a Google Sheets spreadsheet.

### Parameters
- `spreadsheet_id` (str): ID of the spreadsheet to get the metadata from.
"""


class GetSpreadsheetInfoInputSchema(BaseModel):
    spreadsheet_id: str = Field(..., description="ID of the spreadsheet to get the metadata from.")

    model_config = {"arbitrary_types_allowed": True}


class GetSpreadsheetInfo(GoogleSheetsBase):
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "get-spreadsheet-info"
    description: str = DESCRIPTION_GET_SPREADSHEET_INFO
    input_schema: ClassVar[type[BaseModel]] = GetSpreadsheetInfoInputSchema

    def execute(
        self,
        input_data: GetSpreadsheetInfoInputSchema,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Get the comprehensive metadata of a Google Sheets spreadsheet.

        Args:
            input_data (GetSpreadsheetInfoInputSchema): Spreadsheet details.
            config (RunnableConfig | None): Optional runtime config.
            **kwargs: Additional arguments.

        Returns:
            dict[str, Any]: The comprehensive metadata of the spreadsheet.

        Raises:
            ToolExecutionException: If spreadsheet metadata retrieval fails.
        """
        logger.info(
            f"{self.name} ({self.id}) - Getting spreadsheet metadata from spreadsheet {input_data.spreadsheet_id}"
        )
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        try:
            result = self.client.spreadsheets().get(spreadsheetId=input_data.spreadsheet_id).execute()

            logger.info(f"Spreadsheet metadata retrieved from spreadsheet {input_data.spreadsheet_id}: {result}")

            logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")
            return {"content": result}

        except Exception as e:
            raise ToolExecutionException(str(e), recoverable=True)
