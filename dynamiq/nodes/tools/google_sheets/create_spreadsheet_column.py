from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.tools.google_sheets.google_sheets_base import GoogleSheetsBase
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

DESCRIPTION_CREATE_SPREADSHEET_COLUMN = """## Create a Spreadsheet Column
Create a new, empty column in a specified sheet of a Google Sheets spreadsheet at a given index.
Optionally inherit formatting from the column before the new column.

### Parameters
- `spreadsheet_id` (str): ID of the spreadsheet to create the column in.
- `sheet_id` (int): ID of the sheet to create the column in.
- `insert_index` (int, optional): Index of the column to insert the new column at.
    Defaults to 0.
- `inherit_from_before` (bool, optional): Whether to inherit the format and data from the column before the new column.
    Defaults to False.
"""


class CreateSpreadsheetColumnSchema(BaseModel):
    spreadsheet_id: str = Field(..., description="ID of the spreadsheet to create the column in.")
    sheet_id: int = Field(..., description="ID of the sheet to create the column in.")
    insert_index: int = Field(0, description="Index of the column to insert the new column at.")
    inherit_from_before: bool = Field(
        False,
        description="Whether to inherit the format and data from the column before the new column.",
    )

    model_config = {"arbitrary_types_allowed": True}


class CreateSpreadsheetColumn(GoogleSheetsBase):
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "create-spreadsheet-column"
    description: str = DESCRIPTION_CREATE_SPREADSHEET_COLUMN
    input_schema: ClassVar[type[BaseModel]] = CreateSpreadsheetColumnSchema

    def execute(
        self,
        input_data: CreateSpreadsheetColumnSchema,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Creates a new, empty column in a specified sheet of a Google Sheets spreadsheet at a given index.
        Optionally inherits formatting from the column before the new column.

        Args:
            input_data (CreateSpreadsheetColumnSchema): Spreadsheet details.
            config (RunnableConfig | None): Optional runtime config.
            **kwargs: Additional arguments.

        Returns:
            dict[str, Any]: The result of the column creation.

        Raises:
            ToolExecutionException: If column creation fails.
        """
        logger.info(
            f"{self.name} ({self.id}) - Creating column in "
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
                    "dimension": "COLUMNS",
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
                f"Column created in sheet {input_data.sheet_id} "
                f"of spreadsheet {input_data.spreadsheet_id} "
                f"at index {input_data.insert_index}"
            )

            logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")
            return {"content": result}

        except Exception as e:
            raise ToolExecutionException(str(e), recoverable=True)
