from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.tools.google_sheets.google_sheets_base import GoogleSheetsBase
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

DESCRIPTION_BATCH_UPDATE_VALUES = """## Batch Update Values in a Spreadsheet
Update values in a specified range of cells in a Google Sheets spreadsheet
or append values as new rows if `first_cell_location` is not provided.

### Parameters
- `spreadsheet_id` (str): ID of the spreadsheet to update values in.
- `sheet_name` (str): Name of the sheet to update values in.
- `values` (list[list[Any]]): List of values to update in the ranges.
- `first_cell_location` (str, optional): Location of the first cell to update. Defaults to A1.
- `value_input_option` (str, optional): Option for how to input the values into the spreadsheet.
    Defaults to USER_ENTERED.
- `include_values_in_response` (bool, optional): Whether to include the values in the response. Defaults to False.
"""


class BatchUpdateValuesInputSchema(BaseModel):
    spreadsheet_id: str = Field(..., description="ID of the spreadsheet to get values from.")
    sheet_name: str = Field(..., description="Name of the sheet to update values in.")
    values: list[list[Any]] = Field(..., description="List of values to update in the ranges.")
    value_input_option: str = Field(
        "USER_ENTERED", description="Option for how to input the values into the spreadsheet. Defaults to USER_ENTERED."
    )
    first_cell_location: str | None = Field(None, description="Location of the first cell to update in the ranges.")
    include_values_in_response: bool | None = Field(False, description="Whether to include the values in the response.")

    model_config = {"arbitrary_types_allowed": True}


class BatchUpdateValues(GoogleSheetsBase):
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "batch-update-values"
    description: str = DESCRIPTION_BATCH_UPDATE_VALUES
    input_schema: ClassVar[type[BaseModel]] = BatchUpdateValuesInputSchema

    def execute(
        self,
        input_data: BatchUpdateValuesInputSchema,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Updates values in a specified range of cells in a Google Sheets spreadsheet
        or appends values as new rows if `first_cell_location` is not provided.

        Args:
            input_data (BatchUpdateValuesInputSchema): Spreadsheet details.
            config (RunnableConfig | None): Optional runtime config.
            **kwargs: Additional arguments.

        Returns:
            dict[str, Any]: The result of the values update.

        Raises:
            ToolExecutionException: If values update fails.
        """
        if input_data.first_cell_location:
            logger.info(
                f"{self.name} ({self.id}) - Batch updating values "
                f"starting at {input_data.first_cell_location} "
                f"in sheet {input_data.sheet_name} "
                f"of spreadsheet {input_data.spreadsheet_id}"
            )
        else:
            logger.info(
                f"{self.name} ({self.id}) - No first cell location provided, "
                f"batch appending values in sheet {input_data.sheet_name} "
                f"of spreadsheet {input_data.spreadsheet_id}"
            )

        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        print(f"Value input option: {input_data.value_input_option}")

        try:
            if input_data.first_cell_location:
                data = [
                    {
                        "range": f"{input_data.sheet_name}!{input_data.first_cell_location}",
                        "majorDimension": "ROWS",
                        "values": input_data.values,
                    }
                ]
                body = {
                    "valueInputOption": input_data.value_input_option,
                    "data": data,
                    "includeValuesInResponse": input_data.include_values_in_response,
                }
                result = (
                    self.client.spreadsheets()
                    .values()
                    .batchUpdate(spreadsheetId=input_data.spreadsheet_id, body=body)
                    .execute()
                )

                logger.info(
                    f"Values updated in sheet {input_data.sheet_name} of spreadsheet {input_data.spreadsheet_id}"
                )

                logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")
                return {"content": result}

            result = (
                self.client.spreadsheets()
                .values()
                .append(
                    spreadsheetId=input_data.spreadsheet_id,
                    range=f"{input_data.sheet_name}!A1",
                    valueInputOption=input_data.value_input_option,
                    insertDataOption="INSERT_ROWS",
                    body={"values": input_data.values},
                    includeValuesInResponse=input_data.include_values_in_response,
                )
                .execute()
            )
            logger.info(f"Values appended in sheet {input_data.sheet_name} of spreadsheet {input_data.spreadsheet_id}")

            logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")
            return {"content": result}

        except Exception as e:
            raise ToolExecutionException(str(e), recoverable=True)
