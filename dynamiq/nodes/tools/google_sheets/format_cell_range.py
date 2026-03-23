from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.tools.google_sheets.google_sheets_base import GoogleSheetsBase
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

DESCRIPTION_FORMAT_CELL_RANGE = """## Format a Cell Range in a Spreadsheet
Apply a uniform text and background color formatting to a range of cells in a Google Sheets spreadsheet.

### Parameters
- `spreadsheet_id` (str): ID of the spreadsheet to format the cell range in.
- `worksheet_id` (int): ID of the worksheet to format the cell range in.
- `start_row_index` (int): Index of the first row to format.
- `end_row_index` (int): Index of the last row to format.
- `start_column_index` (int): Index of the first column to format.
- `end_column_index` (int): Index of the last column to format.
- `bold` (bool, optional): Whether to bold the cells. Defaults to False.
- `italic` (bool, optional): Whether to italicize the cells. Defaults to False.
- `underline` (bool, optional): Whether to underline the cells. Defaults to False.
- `strikethrough` (bool, optional): Whether to strikethrough the cells. Defaults to False.
- `font_size` (int, optional): Size of the font to apply to the cells. Defaults to 11.
- `red` (float, optional): Red color value to apply to the cells. Defaults to 0.
- `green` (float, optional): Green color value to apply to the cells. Defaults to 0.
- `blue` (float, optional): Blue color value to apply to the cells. Defaults to 0.
"""


class FormatCellRangeInputSchema(BaseModel):
    spreadsheet_id: str = Field(..., description="ID of the spreadsheet to format the cell range in.")
    worksheet_id: int = Field(..., description="ID of the worksheet to format the cell range in.")
    start_row_index: int = Field(..., description="Index of the first row to format.")
    end_row_index: int = Field(..., description="Index of the last row to format.")
    start_column_index: int = Field(..., description="Index of the first column to format.")
    end_column_index: int = Field(..., description="Index of the last column to format.")
    bold: bool | None = Field(None, description="Whether to bold the cells.")
    italic: bool | None = Field(None, description="Whether to italicize the cells.")
    underline: bool | None = Field(None, description="Whether to underline the cells.")
    strikethrough: bool | None = Field(None, description="Whether to strikethrough the cells.")
    font_size: int | None = Field(None, description="Size of the font to apply to the cells.")
    red: float | None = Field(0.0, description="Red color value to apply to the cells.")
    green: float | None = Field(0.0, description="Green color value to apply to the cells.")
    blue: float | None = Field(0.0, description="Blue color value to apply to the cells.")

    model_config = {"arbitrary_types_allowed": True}


class FormatCellRange(GoogleSheetsBase):
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "format-cell-range"
    description: str = DESCRIPTION_FORMAT_CELL_RANGE
    input_schema: ClassVar[type[BaseModel]] = FormatCellRangeInputSchema

    def execute(
        self,
        input_data: FormatCellRangeInputSchema,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Applies a uniform text and background color formatting to a range of cells in a Google Sheets spreadsheet.

        Args:
            input_data (FormatCellRangeInputSchema): Spreadsheet details.
            config (RunnableConfig | None): Optional runtime config.
            **kwargs: Additional arguments.

        Returns:
            dict[str, Any]: The result of the cell range formatting.

        Raises:
            ToolExecutionException: If cell range formatting fails.
        """
        logger.info(
            f"{self.name} ({self.id}) - Formatting cell "
            f"range {input_data.start_row_index}:{input_data.end_row_index}, "
            f"{input_data.start_column_index}:{input_data.end_column_index} "
            f"in worksheet {input_data.worksheet_id} "
            f"of spreadsheet {input_data.spreadsheet_id}"
        )
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        text_format: dict[str, Any] = {
            k: v
            for k, v in {
                "bold": input_data.bold,
                "italic": input_data.italic,
                "underline": input_data.underline,
                "strikethrough": input_data.strikethrough,
                "fontSize": input_data.font_size,
            }.items()
            if v is not None and v is not False
        }

        request = {
            "repeatCell": {
                "range": {
                    "sheetId": input_data.worksheet_id,
                    "startRowIndex": input_data.start_row_index,
                    "endRowIndex": input_data.end_row_index,
                    "startColumnIndex": input_data.start_column_index,
                    "endColumnIndex": input_data.end_column_index,
                },
                "cell": {
                    "userEnteredFormat": {
                        "backgroundColor": {
                            "red": input_data.red,
                            "green": input_data.green,
                            "blue": input_data.blue,
                        },
                        "textFormat": text_format,
                    }
                },
                "fields": "userEnteredFormat(backgroundColor,textFormat)",
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
                f"Cell range formatted in worksheet {input_data.worksheet_id} "
                f"of spreadsheet {input_data.spreadsheet_id}"
            )

            logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")
            return {"content": result}

        except Exception as e:
            raise ToolExecutionException(str(e), recoverable=True)
