from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.tools.google_sheets.google_sheets_base import GoogleSheetsBase
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

DESCRIPTION_CREATE_SPREADSHEET = """## Create a Spreadsheet
Creates an empty Google Sheets spreadsheet with an optional title.

### Parameters
- `title` (str, optional): Title of the spreadsheet. Defaults to "Untitled Spreadsheet".
"""


class CreateSpreadsheetInputSchema(BaseModel):
    title: str | None = Field("Untitled Spreadsheet", description="Title of the spreadsheet.")

    model_config = {"arbitrary_types_allowed": True}


class CreateSpreadsheet(GoogleSheetsBase):
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "create-spreadsheet"
    description: str = DESCRIPTION_CREATE_SPREADSHEET
    input_schema: ClassVar[type[BaseModel]] = CreateSpreadsheetInputSchema

    def execute(
        self,
        input_data: CreateSpreadsheetInputSchema,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Creates an empty Google Sheets spreadsheet with an optional title.

        Args:
            input_data (CreateSpreadsheetInputSchema): Spreadsheet details.
            config (RunnableConfig | None): Optional runtime config.
            **kwargs: Additional arguments.

        Returns:
            dict[str, Any]: The ID of the created spreadsheet.

        Raises:
            ToolExecutionException: If spreadsheet creation fails.
        """
        logger.info(f"{self.name} ({self.id}) - Creating spreadsheet with title {input_data.title}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        spreadsheet = {"properties": {"title": input_data.title}}

        try:
            result = self.client.spreadsheets().create(body=spreadsheet, fields="spreadsheetId").execute()
            logger.info(f"Spreadsheet created with ID: {result.get('spreadsheetId')}")

            logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")
            return {"content": result.get("spreadsheetId")}

        except Exception as e:
            raise ToolExecutionException(str(e), recoverable=True)
