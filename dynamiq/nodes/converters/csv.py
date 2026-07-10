import csv
import enum
from io import BytesIO, StringIO
from typing import Any, ClassVar, Iterator, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dynamiq.components.converters.utils import build_source_metadata, normalize_table_headers
from dynamiq.nodes.node import Node, NodeGroup, RunnableConfig, ensure_config
from dynamiq.types.cancellation import check_cancellation
from dynamiq.utils.logger import logger


class CSVDocumentCreationMode(str, enum.Enum):
    """Supported document boundaries for delimited files."""

    ONE_DOC_PER_ROW = "one-doc-per-row"
    ONE_DOC_PER_FILE = "one-doc-per-file"


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
        metadata_columns (list[str] | None): Column names to extract as metadata for each document.
    """

    file_paths: list[str] | None = Field(
        default=None, description="List of CSV file paths. Either file_paths or files must be provided."
    )
    files: list[BytesIO | bytes] | None = Field(
        default=None, description="List of file objects or bytes representing CSV files."
    )
    delimiter: str | None = Field(
        default=None,
        description="Delimiter used in the CSV files. If not provided, the Node's configured delimiter is used."
    )
    content_column: str | None = Field(
        default=None, description="Name of the column that will be used as the document's main content."
    )
    metadata_columns: list[str] | None = Field(
        default=None,
        description="Optional list of column names to extract as metadata for each document. Can be None.",
    )
    metadata: dict | list | None = Field(
        default=None,
        description="External metadata to be merged with metadata extracted from CSV rows."
    )
    document_creation_mode: CSVDocumentCreationMode | None = Field(
        default=None,
        description="Override the node's CSV document creation mode for this run.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_source(self):
        file_paths, files = self.file_paths, self.files
        if not file_paths and not files:
            raise ValueError("Either `file_paths` or `files` must be provided.")
        return self


class CSVConverter(Node):
    """
    A Node that converts CSV files into a standardized document format.

    The converter can create a self-describing document per row or preserve the
    complete file as plain text. Setting ``content_column`` retains the legacy
    selected-column row behavior.

    Attributes:
        name (str): Display name of the node.
        group (NodeGroup.CONVERTERS): Node group classification.
        delimiter (str | None): Character used to separate fields in the CSV files. Defaults to comma.
        content_column (str | None): Name of the column to use as the main document content.
        metadata_columns (list[str] | None): Column names to extract as metadata for each document.
        input_schema (type[CSVConverterInputSchema]): Schema for validating input parameters.
    """

    name: str = "csv-file-converter"
    group: Literal[NodeGroup.CONVERTERS] = NodeGroup.CONVERTERS
    delimiter: str | None = Field(default=",", description="Delimiter used in the CSV files.")
    document_creation_mode: CSVDocumentCreationMode = Field(
        default=CSVDocumentCreationMode.ONE_DOC_PER_ROW,
        description="Create one self-describing document per row or preserve the whole file as plain text.",
    )
    content_column: str | None = Field(
        default=None,
        description="Column used as content. When omitted, every row is rendered as header-value text.",
    )
    metadata_columns: list[str] | None = Field(
        default=None,
        description="Optional list of column names to extract as metadata for each document. Can be None.",
    )
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
        delimiter = input_data.delimiter or self.delimiter or ","
        content_column = input_data.content_column or self.content_column
        metadata_columns = input_data.metadata_columns or self.metadata_columns
        document_creation_mode = input_data.document_creation_mode or self.document_creation_mode
        source_index = 0

        if input_data.file_paths:
            for path in input_data.file_paths:
                check_cancellation(config)
                try:
                    with open(path, encoding="utf-8-sig", newline="") as csv_file:
                        external_metadata = self._metadata_for_source(input_data.metadata, source_index, total_files)
                        for doc in self._process_text(
                            csv_file.read(),
                            path,
                            delimiter,
                            document_creation_mode,
                            content_column,
                            metadata_columns,
                            external_metadata,
                        ):
                            all_documents.append(doc)
                        success_count += 1
                except Exception as e:
                    logger.error(f"Error processing file {path}: {str(e)}")
                    last_error = e
                finally:
                    source_index += 1

        if input_data.files:
            for file_obj in input_data.files:
                check_cancellation(config)
                source_name = getattr(file_obj, "name", "in-memory file")
                try:
                    if isinstance(file_obj, bytes):
                        file_obj = BytesIO(file_obj)

                    external_metadata = self._metadata_for_source(input_data.metadata, source_index, total_files)
                    file_obj.seek(0)
                    text = file_obj.read().decode("utf-8-sig", errors="replace")
                    for doc in self._process_text(
                        text,
                        source_name,
                        delimiter,
                        document_creation_mode,
                        content_column,
                        metadata_columns,
                        external_metadata,
                    ):
                        all_documents.append(doc)
                    success_count += 1
                except Exception as e:
                    logger.error(f"Error processing file {source_name}: {str(e)}")
                    last_error = e
                finally:
                    source_index += 1

        if success_count == 0 and last_error is not None:
            raise last_error

        if success_count < total_files:
            logger.warning(f"Processed {success_count} out of {total_files} files successfully")

        return {"documents": all_documents}

    def _process_text(
        self,
        text: str,
        source: str,
        delimiter: str,
        document_creation_mode: CSVDocumentCreationMode,
        content_column: str | None,
        metadata_columns: list[str] | None,
        external_metadata: dict | None,
    ) -> Iterator[dict]:
        reader = csv.reader(StringIO(text), delimiter=delimiter, strict=True)
        if document_creation_mode == CSVDocumentCreationMode.ONE_DOC_PER_FILE:
            rows = list(reader)
            if not rows:
                return
            metadata = self._build_metadata(source, external_metadata)
            metadata.update(
                {
                    "content_type": self._content_type(source, delimiter),
                    "document_type": "table",
                    "row_count": max(0, len(rows) - 1),
                }
            )
            yield {"content": text.strip(), "metadata": metadata}
            return
        yield from self._process_reader(
            reader,
            source,
            content_column,
            metadata_columns,
            external_metadata,
            content_type=self._content_type(source, delimiter),
        )

    def _process_reader(
        self,
        reader: Iterator[list[str]],
        source: str,
        content_column: str | None,
        metadata_columns: list[str] | None,
        external_metadata: dict | None,
        content_type: str = "text/csv",
    ) -> Iterator[dict]:
        if content_column:
            yield from self._process_named_rows(reader, source, content_column, metadata_columns, external_metadata)
            return
        yield from self._process_all_columns(reader, source, external_metadata, content_type)

    def _process_named_rows(
        self,
        reader: Iterator[list[str]],
        source: str,
        content_column: str,
        metadata_columns: list[str] | None,
        external_metadata: dict | None,
    ) -> Iterator[dict]:
        try:
            headers = next(reader)
        except StopIteration:
            return
        dict_reader = (dict(zip(headers, row)) for row in reader)
        yield from self._process_rows_generator(
            dict_reader, source, content_column, metadata_columns, external_metadata
        )

    def _process_all_columns(
        self,
        reader: Iterator[list[str]],
        source: str,
        external_metadata: dict | None,
        content_type: str = "text/csv",
    ) -> Iterator[dict]:
        try:
            raw_headers = next(reader)
        except StopIteration:
            return

        for row_number, row in enumerate(reader, start=2):
            width = max(len(raw_headers), len(row))
            headers = normalize_table_headers(raw_headers, width)
            padded = row + [""] * (width - len(row))
            fields = [f"{headers[index]}: {value}" for index, value in enumerate(padded) if value.strip()]
            if not fields:
                continue
            metadata = self._build_metadata(source, external_metadata)
            metadata.update({"content_type": content_type, "document_type": "table_row", "row_number": row_number})
            yield {"content": "\n".join(fields), "metadata": metadata}

    @staticmethod
    def _content_type(source: str, delimiter: str) -> str:
        return "text/tab-separated-values" if delimiter == "\t" or source.lower().endswith(".tsv") else "text/csv"

    @staticmethod
    def _metadata_for_source(metadata: dict | list | None, index: int, total: int) -> dict | None:
        if metadata is None or isinstance(metadata, dict):
            return metadata
        if len(metadata) != total:
            raise ValueError(f"The metadata list length [{len(metadata)}] must match the file count [{total}].")
        return metadata[index]

    @staticmethod
    def _build_metadata(source: str, external_metadata: dict | None) -> dict:
        return build_source_metadata(external_metadata, source)

    def _process_rows_generator(
        self,
        reader: Iterator[dict[str, str]],
        source: str,
        content_column: str,
        metadata_columns: list[str] | None,
        external_metadata: dict | list | None,

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
            metadata_columns (list[str]|None): Column names to include as metadata in the
                resulting document dictionary.
            external_metadata (dict | list | None): External metadata to merge with CSV metadata.
                If a key exists in both, the CSV metadata will override the external metadata.


        Yields:
            dict: A document dictionary with two keys:
                - 'content': The content extracted from the specified `content_column`.
                - 'metadata': A dict containing merged metadata from CSV and external sources,
                  plus a 'source' key identifying the file or in-memory data source.

        Raises:
            KeyError: If the specified `content_column` is not present in a row of the CSV.
        """
        metadata_columns = metadata_columns or []
        for index, row in enumerate(reader):
            if content_column not in row:
                raise KeyError(
                    f"Content column '{content_column}' not found in CSV " f"(source: {source}) at row {index}"
                )

            merged_metadata = self._build_metadata(source, external_metadata)
            merged_metadata.update({col: row[col] for col in metadata_columns if col in row})
            merged_metadata["row_number"] = index + 2

            yield {
                "content": row[content_column],
                "metadata": merged_metadata,
            }
