from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.tools.google_sheets.create_spreadsheet import CreateSpreadsheet
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

DESCRIPTION_CREATE_SPREADSHEET_FROM_JSON = """## Create a Spreadsheet from JSON
Create a spreadsheet in Google Sheets from a JSON object (a list of homogeneous dictionaries).
The keys of the first dictionary will be used as the headers of the spreadsheet.

### Parameters
- `title` (str, optional): Title of the spreadsheet to create. Defaults to "Untitled Spreadsheet".
- `sheet_name` (str, optional): Name of the sheet to create. Defaults to "Sheet1".
- `json_data` (list[list[Any]]): JSON data to create the spreadsheet from.
"""


class CreateSpreadsheetFromJSONInputSchema(BaseModel):
    title: str | None = Field("Untitled Spreadsheet", description="Title of the spreadsheet to create.")
    sheet_name: str | None = Field("Sheet1", description="Name of the sheet to create.")
    json_data: list[dict[str, Any]] = Field(..., description="JSON data to create the spreadsheet from.")

    model_config = {"arbitrary_types_allowed": True}


class CreateSpreadsheetFromJSON(CreateSpreadsheet):
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "create-spreadsheet-from-json"
    description: str = DESCRIPTION_CREATE_SPREADSHEET_FROM_JSON
    input_schema: ClassVar[type[BaseModel]] = CreateSpreadsheetFromJSONInputSchema

    def execute(
        self,
        input_data: CreateSpreadsheetFromJSONInputSchema,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Creates a spreadsheet in Google Sheets from a JSON object (a list of homogeneous dictionaries).
        The keys of the first dictionary will be used as the headers of the spreadsheet.

        Args:
            input_data (GetSpreadsheetInfoInputSchema): Spreadsheet details.
            config (RunnableConfig | None): Optional runtime config.
            **kwargs: Additional arguments.

        Returns:
            dict[str, Any]: The result of the spreadsheet creation.

        Raises:
            ToolExecutionException: If spreadsheet creation fails.
        """
        logger.info(
            f"{self.name} ({self.id}) - Creating spreadsheet {input_data.title} "
            f"with sheet {input_data.sheet_name} from JSON data"
        )
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        if not input_data.json_data:
            raise ToolExecutionException("No JSON data provided", recoverable=True)

        headers = list(input_data.json_data[0].keys())
        data_rows = [[record.get(h, "") for h in headers] for record in input_data.json_data]

        sheet_request = {
            "properties": {
                "title": input_data.title,
            },
            "sheets": [
                {
                    "properties": {
                        "title": input_data.sheet_name,
                    }
                }
            ],
        }

        try:
            create_spreadsheet_result = (
                self.client.spreadsheets().create(body=sheet_request, fields="spreadsheetId,spreadsheetUrl").execute()
            )

            sheet_id = create_spreadsheet_result.get("spreadsheetId")
            sheet_url = create_spreadsheet_result.get("spreadsheetUrl")

            if not sheet_id or not sheet_url:
                raise ToolExecutionException("Spreadsheet creation failed", recoverable=True)

            logger.info(f"Spreadsheet created with ID: {sheet_id} and URL: {sheet_url}")

            sheet_name = input_data.sheet_name

            data = [
                {
                    "range": f"{sheet_name}!A1",
                    "majorDimension": "ROWS",
                    "values": [headers] + data_rows,
                }
            ]
            body = {
                "valueInputOption": "USER_ENTERED",
                "data": data,
                "includeValuesInResponse": True,
            }
            result = self.client.spreadsheets().values().batchUpdate(spreadsheetId=sheet_id, body=body).execute()

            logger.info(f"Values updated in sheet {sheet_name} of spreadsheet {sheet_id}")

            logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")

            return {"content": result}

        except Exception as e:
            raise ToolExecutionException(str(e), recoverable=True)
