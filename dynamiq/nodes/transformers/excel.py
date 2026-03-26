from io import BytesIO
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.nodes.node import Node, ensure_config
from dynamiq.nodes.types import NodeGroup
from dynamiq.runnables import RunnableConfig


class ExcelToListTransformerInputSchema(BaseModel):
    value: str | BytesIO = Field(..., description="Parameter to provide excel data to transform.")
    sheet_name: str = Field(None, description="Name of the Excel sheet to read")
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ExcelToList(Node):
    group: Literal[NodeGroup.TRANSFORMERS] = NodeGroup.TRANSFORMERS
    name: str = "ExcelToList"
    description: str = "Node that transforms excel file into a list of dictionaries"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_schema: ClassVar[type[ExcelToListTransformerInputSchema]] = ExcelToListTransformerInputSchema

    def execute(
        self, input_data: ExcelToListTransformerInputSchema, config: RunnableConfig = None, **kwargs
    ) -> dict[str, Any]:
        """
        Transform an Excel file into a list of dictionaries.

        Args:
            input_data (ExcelToListTransformerInputSchema): input data for the tool, which includes the data from
                Excel file.
            config (RunnableConfig, optional): Configuration for the runnable, including callbacks.
            **kwargs: Additional arguments passed to the execution context.

        Returns:
            dict[str, Any]: A dictionary containing list of data from Excel file.
        """
        import pandas as pd

        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        value = input_data.value
        sheet_name = input_data.sheet_name
        try:
            if sheet_name is None:
                excel_data = pd.read_excel(value, sheet_name=None)
                result = {sheet: df.to_dict(orient="records") for sheet, df in excel_data.items()}
            else:
                df = pd.read_excel(value, sheet_name=sheet_name)
                result = {sheet_name: df.to_dict(orient="records")}

            return {"content": result}
        except Exception as e:
            raise ValueError(f"Encountered an error while performing transformation. \nError details: {e}")
