import csv
from io import BytesIO, TextIOWrapper
from typing import Any, ClassVar, Iterator, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dynamiq.nodes.node import Node, NodeGroup, RunnableConfig, ensure_config
from dynamiq.utils.logger import logger


class CSVConverterInputSchema(BaseModel):
    """
    Schema defining the input parameters for CSV file conversion.

    This model validates and structures the input configuration for converting CSV files
    into a standardized document format. It ensures that either file paths or file objects
    are provided, along with the necessary column specifications.

    Attributes:
        file_paths (list[str] | None): List of paths to CSV files on the filesystem.
        files (list[BytesIO | bytes] | None): List of file objects or bytes containing CSV data.
        delimiter (str): Character used to separate fields in the CSV files. Defaults to comma.
        content_column (str): Name of the column to use as the main document content.
        metadata_columns (list[str]): Column names to extract as metadata for each document.
    """

    file_paths: list[str] | None = Field(
        default=None, description="List of CSV file paths. Either file_paths or files must be provided."
    )
    files: list[BytesIO | bytes] | None = Field(
        default=None, description="List of file objects or bytes representing CSV files."
    )
    delimiter: str = Field(default=",", description="Delimiter used in the CSV files.")
    content_column: str = Field(..., description="Name of the column that will be used as the document's main content.")
    metadata_columns: list[str] = Field(
        default_factory=list, description="List of column names to extract as metadata for each document."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_source(cls, values):
        file_paths, files = values.file_paths, values.files
        if not file_paths and not files:
            raise ValueError("Either `file_paths` or `files` must be provided.")
        return values


class CSVConverter(Node):
    """
    A Node that converts CSV files into a standardized document format.

    This converter processes CSV files from either file paths or file objects,
    extracting specified content and metadata columns into a structured document format.
    Each row in the CSV becomes a document with content from the specified content column
    and metadata from the specified metadata columns.

    Attributes:
        name (str): Display name of the node.
        group (NodeGroup.CONVERTERS): Node group classification.
        input_schema (type[CSVConverterInputSchema]): Schema for validating input parameters.
    """

    name: str = "CSV File Converter"
    group: Literal[NodeGroup.CONVERTERS] = NodeGroup.CONVERTERS
    input_schema: ClassVar[type[CSVConverterInputSchema]] = CSVConverterInputSchema

    def execute(
        self, input_data: CSVConverterInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, list[Any]]:
        """
        Executes the CSV conversion process.

        Processes one or more CSV files according to the input configuration,
        converting each row into a document with specified content and metadata.
        If some files fail to process but at least one succeeds, logs errors and continues.
        If all files fail, raises the last encountered error.

        Args:
            input_data (CSVConverterInputSchema): Validated input parameters for the conversion.
            config (RunnableConfig | None): Optional runtime configuration.
            **kwargs: Additional keyword arguments passed to execution callbacks.

        Returns:
            dict[str, list[Any]]: Dictionary containing the list of processed documents
                under the 'documents' key.

        Raises:
            Exception: If there are errors reading or processing the CSV files.
        """
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        all_documents = []
        last_error = None
        success_count = 0
        total_files = len(input_data.file_paths or []) + len(input_data.files or [])

        if input_data.file_paths:
            for path in input_data.file_paths:
                try:
                    with open(path, encoding="utf-8") as csv_file:
                        reader = csv.DictReader(csv_file, delimiter=input_data.delimiter)
                        for doc in self._process_rows_generator(
                            reader,
                            source=path,
                            content_column=input_data.content_column,
                            metadata_columns=input_data.metadata_columns,
                        ):
                            all_documents.append(doc)
                        success_count += 1
                except Exception as e:
                    logger.error(f"Error processing file {path}: {str(e)}")
                    last_error = e

        if input_data.files:
            for file_obj in input_data.files:
                source_name = getattr(file_obj, "name", "in-memory file")
                try:
                    if isinstance(file_obj, bytes):
                        file_obj = BytesIO(file_obj)

                    file_text = TextIOWrapper(file_obj, encoding="utf-8")
                    reader = csv.DictReader(file_text, delimiter=input_data.delimiter)

                    for doc in self._process_rows_generator(
                        reader,
                        source=source_name,
                        content_column=input_data.content_column,
                        metadata_columns=input_data.metadata_columns,
                    ):
                        all_documents.append(doc)
                    success_count += 1
                except Exception as e:
                    logger.error(f"Error processing file {source_name}: {str(e)}")
                    last_error = e

        if success_count == 0 and last_error is not None:
            raise last_error

        if success_count < total_files:
            logger.warning(f"Processed {success_count} out of {total_files} files successfully")

        return {"documents": all_documents}

    def _process_rows_generator(
        self,
        reader: csv.DictReader,
        source: str,
        content_column: str,
        metadata_columns: list[str],
    ) -> Iterator[dict]:
        """
        Processes CSV rows into structured document dictionaries in a streaming fashion.

        This method reads CSV rows one by one from the given DictReader, extracts the
        specified content and metadata columns, and yields each row as a dictionary
        with 'content' and 'metadata' fields.

        Args:
            reader (csv.DictReader): The CSV DictReader from which rows are read.
            source (str): The source identifier for the CSV data, e.g., a file path
                or name for in-memory data.
            content_column (str): The name of the column containing the main document content.
            metadata_columns (list[str]): Column names to include as metadata in the
                resulting document dictionary.

        Yields:
            dict: A document dictionary with two keys:
                - 'content': The content extracted from the specified `content_column`.
                - 'metadata': A dict containing metadata from `metadata_columns`,
                  plus a 'source' key identifying the file or in-memory data source.

        Raises:
            KeyError: If the specified `content_column` is not present in a row of the CSV.
        """
        for index, row in enumerate(reader):
            if content_column not in row:
                raise KeyError(
                    f"Content column '{content_column}' not found in CSV " f"(source: {source}) at row {index}"
                )

            metadata = {col: row[col] for col in metadata_columns if col in row}
            metadata["source"] = source

            yield {
                "content": row[content_column],
                "metadata": metadata,
            }
