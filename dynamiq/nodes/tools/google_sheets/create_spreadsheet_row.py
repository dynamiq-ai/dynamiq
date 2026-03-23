from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.tools.google_sheets.google_sheets_base import GoogleSheetsBase
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

DESCRIPTION_CREATE_SPREADSHEET_ROW = """## Create a Spreadsheet Row
Insert a new, empty row into a specified sheet of a Google Sheets spreadsheet at a given index.
Optionally inherit formatting from the row above.

### Parameters
- `spreadsheet_id` (str): ID of the spreadsheet to create the row in.
- `sheet_id` (int): ID of the sheet to create the row in.
- `insert_index` (int, optional): Index of the row to insert the new row at. Defaults to 0.
- `inherit_from_before` (bool, optional): Whether to inherit the format and data from the row before the new row.
    Defaults to False.
"""


class CreateSpreadsheetRowSchema(BaseModel):
    spreadsheet_id: str = Field(..., description="ID of the spreadsheet to create the row in.")
    sheet_id: int = Field(..., description="ID of the sheet to create the row in.")
    insert_index: int | None = Field(0, description="Index of the row to insert the new row at.")
    inherit_from_before: bool | None = Field(
        False,
        description="Whether to inherit the format and data from the row before the new row.",
    )

    model_config = {"arbitrary_types_allowed": True}


class CreateSpreadsheetRow(GoogleSheetsBase):
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "create-spreadsheet-row"
    description: str = DESCRIPTION_CREATE_SPREADSHEET_ROW
    input_schema: ClassVar[type[BaseModel]] = CreateSpreadsheetRowSchema

    def execute(
        self,
        input_data: CreateSpreadsheetRowSchema,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Inserts a new, empty row into a specified sheet of a Google Sheets spreadsheet at a given index.
        Optionally inherits formatting from the row above.

        Args:
            input_data (CreateSpreadsheetRowSchema): Spreadsheet details.
            config (RunnableConfig | None): Optional runtime config.
            **kwargs: Additional arguments.

        Returns:
            dict[str, Any]: The result of the row creation.

        Raises:
            ToolExecutionException: If row creation fails.
        """
        logger.info(
            f"{self.name} ({self.id}) - Inserting row in "
            f"sheet {input_data.sheet_id} "
            f"of spreadsheet {input_data.spreadsheet_id} "
            f"at index {input_data.insert_index}"
        )
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        request = {
            "insertDimension": {
                "range": {
                    "sheetId": input_data.sheet_id,
                    "dimension": "ROWS",
                    "startIndex": input_data.insert_index,
                    "endIndex": input_data.insert_index + 1,
                },
                "inheritFromBefore": input_data.inherit_from_before,
            }
        }

        try:
            result = (
                self.client.spreadsheets()
                .batchUpdate(
                    spreadsheetId=input_data.spreadsheet_id,
                    body={"requests": [request]},
                )
                .execute()
            )
            logger.info(
                f"Row inserted in sheet {input_data.sheet_id} "
                f"of spreadsheet {input_data.spreadsheet_id} "
                f"at index {input_data.insert_index}"
            )

            logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")
            return {"content": result}

        except Exception as e:
            raise ToolExecutionException(str(e), recoverable=True)
