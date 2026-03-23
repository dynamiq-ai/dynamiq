from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.tools.google_sheets.google_sheets_base import GoogleSheetsBase
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

DESCRIPTION_BATCH_GET = """## Batch Get Values from a Spreadsheet
Retrieve the data from one or more specified cell ranges in a Google Sheets spreadsheet.

### Parameters
- `spreadsheet_id` (str): ID of the spreadsheet to get values from.
- `ranges` (list[str], optional): List of ranges to get values from. Defaults to the range A1:ZZ (whole spreadsheet).
"""


class BatchGetValuesInputSchema(BaseModel):
    spreadsheet_id: str = Field(..., description="ID of the spreadsheet to get values from.")
    ranges: list[str] | None = Field(
        "A1:ZZ", description="List of ranges to get values from. Defaults to the range A1:ZZ (whole spreadsheet)."
    )

    model_config = {"arbitrary_types_allowed": True}


class BatchGetValues(GoogleSheetsBase):
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "batch-get-values"
    description: str = DESCRIPTION_BATCH_GET
    input_schema: ClassVar[type[BaseModel]] = BatchGetValuesInputSchema

    def execute(
        self,
        input_data: BatchGetValuesInputSchema,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Retrieves the data from one or more specified cell ranges in a Google Sheets spreadsheet.

        Args:
            input_data (BatchGetValuesInputSchema): Spreadsheet details.
            config (RunnableConfig | None): Optional runtime config.
            **kwargs: Additional arguments.

        Returns:
            dict[str, Any]: The data from the specified cell ranges in the spreadsheet.

        Raises:
            ToolExecutionException: If values retrieval fails.
        """
        logger.info(
            f"{self.name} ({self.id}) - Batch retrieving values "
            f"from ranges {input_data.ranges} of spreadsheet {input_data.spreadsheet_id}"
        )
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        try:
            result = (
                self.client.spreadsheets()
                .values()
                .batchGet(
                    spreadsheetId=input_data.spreadsheet_id,
                    ranges=input_data.ranges,
                )
                .execute()
            )
            logger.info(f"Values retrieved from ranges {input_data.ranges} of spreadsheet {input_data.spreadsheet_id}")

            logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")
            return {"content": result}

        except Exception as e:
            raise ToolExecutionException(str(e), recoverable=True)
