from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.tools.google_sheets.google_sheets_base import GoogleSheetsBase
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

DESCRIPTION_FIND_WORKSHEET_BY_TITLE = """## Find a Worksheet by Title in a Spreadsheet
Find a worksheet by title in a Google Sheets spreadsheet.

### Parameters
- `spreadsheet_id` (str): ID of the spreadsheet to find the worksheet in.
- `title` (str): Title of the worksheet to find.
"""


class FindWorksheetByTitleInputSchema(BaseModel):
    spreadsheet_id: str = Field(..., description="ID of the spreadsheet to find the worksheet in.")
    title: str = Field(..., description="Title of the worksheet to use to find the worksheet.")

    model_config = {"arbitrary_types_allowed": True}


class FindWorksheetByTitle(GoogleSheetsBase):
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "find-worksheet-by-title"
    description: str = DESCRIPTION_FIND_WORKSHEET_BY_TITLE
    input_schema: ClassVar[type[BaseModel]] = FindWorksheetByTitleInputSchema

    def execute(
        self,
        input_data: FindWorksheetByTitleInputSchema,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Finds a worksheet by title in a Google Sheets spreadsheet.

        Args:
            input_data (FindWorksheetByTitleInputSchema): Spreadsheet details.
            config (RunnableConfig | None): Optional runtime config.
            **kwargs: Additional arguments.

        Returns:
            dict[str, Any]: The worksheet with the given title.

        Raises:
            ToolExecutionException: If worksheet search fails.
        """
        logger.info(
            f"{self.name} ({self.id}) - Searching for worksheet with "
            f"title {input_data.title} in spreadsheet {input_data.spreadsheet_id}"
        )
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        try:
            metadata = self.client.spreadsheets().get(spreadsheetId=input_data.spreadsheet_id).execute()

            result = None

            for sheet in metadata.get("sheets", []):
                if sheet.get("properties", {}).get("title") == input_data.title:
                    result = sheet
                    break

            if result is None:
                raise ToolExecutionException(
                    f"No worksheet found with title {input_data.title} in spreadsheet {input_data.spreadsheet_id}"
                )

            logger.info(f"Found worksheet with title {input_data.title} in spreadsheet {input_data.spreadsheet_id}")

            logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")
            return {"content": result}

        except Exception as e:
            raise ToolExecutionException(str(e), recoverable=True)
