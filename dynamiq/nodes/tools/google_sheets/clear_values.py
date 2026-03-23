from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.tools.google_sheets.google_sheets_base import GoogleSheetsBase
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

DESCRIPTION_CLEAR_VALUES = """## Clear Values in a Spreadsheet
Clear values in a specified range of cells in a Google Sheets spreadsheet.

### Parameters
- `spreadsheet_id` (str): ID of the spreadsheet to clear values in.
- `range` (str): Range of the values to clear.
"""


class ClearValuesInputSchema(BaseModel):
    spreadsheet_id: str = Field(..., description="ID of the spreadsheet to clear values in.")
    range: str = Field(..., description="Range of the values to clear.")

    model_config = {"arbitrary_types_allowed": True}


class ClearValues(GoogleSheetsBase):
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "clear-values"
    description: str = DESCRIPTION_CLEAR_VALUES
    input_schema: ClassVar[type[BaseModel]] = ClearValuesInputSchema

    def execute(
        self,
        input_data: ClearValuesInputSchema,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Clears values in a specified range of cells in a Google Sheets spreadsheet.

        Args:
            input_data (ClearValuesInputSchema): Spreadsheet details.
            config (RunnableConfig | None): Optional runtime config.
            **kwargs: Additional arguments.

        Returns:
            dict[str, Any]: The result of the values clearing.

        Raises:
            ToolExecutionException: If clearing values fails.
        """
        logger.info(
            f"{self.name} ({self.id}) - Clearing values "
            f"in range {input_data.range} "
            f"of spreadsheet {input_data.spreadsheet_id}"
        )
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        try:
            result = (
                self.client.spreadsheets()
                .values()
                .clear(
                    spreadsheetId=input_data.spreadsheet_id,
                    range=input_data.range,
                    body={},
                )
                .execute()
            )
            logger.info(f"Values cleared in range {input_data.range} of spreadsheet {input_data.spreadsheet_id}")

            logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")
            return {"content": result}

        except Exception as e:
            raise ToolExecutionException(str(e), recoverable=True)
