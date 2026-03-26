import csv
import json
from io import BytesIO, StringIO
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.nodes.node import Node, ensure_config
from dynamiq.nodes.types import NodeGroup
from dynamiq.runnables import RunnableConfig


class CSVToListTransformerInputSchema(BaseModel):
    value: str | BytesIO = Field(..., description="Parameter to provide value to transform.")
    is_header_exists: bool = Field(
        True, description="Parameter to specify whether the first row contains column headers."
    )
    delimiter: str = Field(",", description="Parameter to specify the separator between columns.")
    model_config = ConfigDict(arbitrary_types_allowed=True)


class CSVToList(Node):
    group: Literal[NodeGroup.TRANSFORMERS] = NodeGroup.TRANSFORMERS
    name: str = "CsvToList"
    description: str = "Node that transforms csv to list of data in object format"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_schema: ClassVar[type[CSVToListTransformerInputSchema]] = CSVToListTransformerInputSchema

    def execute(
        self, input_data: CSVToListTransformerInputSchema, config: RunnableConfig = None, **kwargs
    ) -> dict[str, Any]:
        """
        Transform data from csv file to list of data in object format.

        Args:
            input_data (CSVToListTransformerInputSchema): input data for the tool, which includes csv value
                to transform.
            config (RunnableConfig, optional): Configuration for the runnable, including callbacks.
            **kwargs: Additional arguments passed to the execution context.

        Returns:
            dict[str, Any]: A dictionary containing data from csv file as list of dictionaries.
        """
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        value = input_data.value
        header = input_data.is_header_exists
        delimiter = input_data.delimiter
        try:
            if isinstance(value, BytesIO):
                value = value.getvalue().decode("utf-8")
            csv_file = StringIO(value)

            if header:
                csv_reader = csv.DictReader(csv_file, delimiter=delimiter)
            else:
                csv_reader = csv.reader(csv_file, delimiter=delimiter)
                csv_reader = ({f"column{i}": value for i, value in enumerate(row)} for row in csv_reader)

            result = [row for row in csv_reader]
            return {"content": result}
        except Exception as e:
            raise ValueError(f"Encountered an error while performing transformation. \nError details: {e}")


class JSONToCSVTransformerInputSchema(BaseModel):
    value: str = Field(..., description="Parameter to provide value to transform.")


class JSONToCSV(Node):
    group: Literal[NodeGroup.TRANSFORMERS] = NodeGroup.TRANSFORMERS
    name: str = "JSONToCsv"
    description: str = "Node that transforms JSON to csv"

    input_schema: ClassVar[type[JSONToCSVTransformerInputSchema]] = JSONToCSVTransformerInputSchema

    def execute(
        self, input_data: JSONToCSVTransformerInputSchema, config: RunnableConfig = None, **kwargs
    ) -> dict[str, Any]:
        """
        Transform JSON to format for csv file.

        Args:
            input_data (JSONToCSVTransformerInputSchema): input data for the tool, which includes the JSON value.
            config (RunnableConfig, optional): Configuration for the runnable, including callbacks.
            **kwargs: Additional arguments passed to the execution context.

        Returns:
            dict[str, Any]: A dictionary containing JSON data in BytesIO format for csv file.
        """
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        value = input_data.value
        try:
            value = json.loads(value)
            if not isinstance(value, list) or not value:
                raise ValueError("Invalid data format: Expected a non-empty list")

            if not isinstance(value[0], dict):
                raise ValueError("Invalid data format: No valid field names found in the list.")
            fieldnames = list(value[0].keys())

            csv_string_io = StringIO()
            csv_writer = csv.DictWriter(csv_string_io, fieldnames=fieldnames)
            csv_writer.writeheader()
            csv_writer.writerows(value)

            result = BytesIO(csv_string_io.getvalue().encode("utf-8"))
            return {"content": result}
        except Exception as e:
            raise ValueError(f"Encountered an error while performing transformation. \nError details: {e}")
