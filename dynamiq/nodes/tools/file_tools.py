import json
import logging
import os
import re
from io import BytesIO
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.converters import (
    DOCXFileConverter,
    HTMLConverter,
    LLMImageConverter,
    PPTXFileConverter,
    PyPDFConverter,
    TextFileConverter,
)
from dynamiq.nodes.converters.pypdf import DocumentCreationMode as PyPDFDocumentCreationMode
from dynamiq.nodes.llms.base import BaseLLM
from dynamiq.nodes.node import ensure_config
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.storages.file.base import FileStore
from dynamiq.utils.file_types import EXTENSION_MAP, FileType

logger = logging.getLogger(__name__)

EXTRACTED_TEXT_SUFFIX = ".extracted.txt"


def detect_file_type(file: BytesIO, filename: str) -> FileType | None:
    """
    Detect the file type based on file extension.

    Args:
        file: The file object (BytesIO)
        filename: The filename to extract extension from

    Returns:
        FileType: The detected file type, or None if not found
    """
    try:
        if not filename and hasattr(file, "name"):
            filename = file.name

        if not filename:
            logger.warning("No filename provided for file type detection")
            return None

        file_ext = os.path.splitext(filename)[1][1:] if filename else ""
        file_ext = file_ext.lower()

        if not file_ext:
            logger.warning(f"No file extension found in filename: {filename}")
            return None

        for file_type, extensions in EXTENSION_MAP.items():
            if file_ext in extensions:
                logger.info(f"Detected file type: {file_type} for file: {filename}")
                return file_type

        logger.warning(f"Unknown file extension: {file_ext} for file: {filename}")
        return None

    except Exception as e:
        logger.warning(f"File type detection failed for {filename}: {str(e)}")
        return None


DEFAULT_FILE_TYPE_TO_CONVERTER_CLASS_MAP = {
    FileType.PDF: PyPDFConverter,
    FileType.DOCX_DOCUMENT: DOCXFileConverter,
    FileType.PPTX_PRESENTATION: PPTXFileConverter,
    FileType.SPREADSHEET: TextFileConverter,
    FileType.HTML: HTMLConverter,
    FileType.TEXT: TextFileConverter,
    FileType.MARKDOWN: TextFileConverter,
    FileType.IMAGE: LLMImageConverter,
}


class FileReadInputSchema(BaseModel):
    """Schema for file read input parameters."""

    file_path: str = Field(default="", description="Path of the file to read")
    instructions: str | None = Field(
        default=None,
        description="Instructions for the file read. If not provided, the file will be read in its entirety.",
    )
    mode: Literal["auto", "full", "chunked", "summary"] = Field(
        default="auto",
        description=(
            "Controls how the file content is returned. "
            "'auto' uses default heuristics, 'full' forces entire content, "
            "'chunked' always returns segmented chunks, and 'summary' returns a short preview."
        ),
    )
    chunk_size_override: int | None = Field(
        default=None,
        description="Optional chunk size override ("
        "in bytes/chars depending on content type) when returning chunked output.",
    )
    max_preview_bytes: int | None = Field(
        default=None,
        description="Optional maximum number of bytes/chars to include when returning summary previews.",
    )
    document_mode: Literal["file", "page"] = Field(
        default="file",
        description="For PDF-like documents, 'page' keeps content separated per page (with metadata).",
    )


class FileWriteInputSchema(BaseModel):
    """Schema for file write input parameters."""

    file_path: str = Field(..., description="Path where the file should be written")
    content: Any = Field(..., description="File content (string, bytes, or structured data for JSON)")
    content_type: str | None = Field(default=None, description="MIME type (auto-detected if not provided)")
    metadata: dict[str, Any] | None = Field(default=None, description="Additional metadata for the file")
    overwrite: bool = Field(
        default=True,
        description="Whether to overwrite the file if it already exists. Defaults to True.",
    )
    content_format: Literal["auto", "text", "json", "binary"] = Field(
        default="auto",
        description=(
            "Hints how to encode the provided content. 'auto' infers from the value, "
            "'text' treats the payload as UTF-8 text, 'json' serializes with json.dumps, "
            "and 'binary' writes raw bytes."
        ),
    )
    encoding: str = Field(default="utf-8", description="Encoding to use when writing textual content.")
    append: bool = Field(
        default=False,
        description="If True, append to the existing file instead of replacing it. "
        "Falls back to overwrite when file is missing.",
    )


class FileReadTool(Node):
    """
    A tool for reading files from storage with intelligent file processing.

    This tool can be passed to Agents to read files from the configured storage backend.
    It automatically detects file types and processes them using appropriate converters to extract text content.
    For large files, it automatically returns chunked content showing first, middle, and last parts.
    For images and PDFs with instructions, uses LLM processing if the model supports vision/PDF input.

    Attributes:
        group (Literal[NodeGroup.TOOLS]): The group to which this tool belongs.
        name (str): The name of the tool.
        description (str): A brief description of the tool.
        file_store (FileStore): File storage to read from.
        llm (BaseLLM): LLM that will be used to process files.
        max_size (int): Maximum size in bytes before chunking (default: 10000).
        chunk_size (int): Size of each chunk in bytes (default: 1000).
        converter_mapping (dict[FileType, Node]): Mapping of file types to converters.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "FileReadTool"
    description: str = """
        Reads files from storage based on the provided file path with intelligent file processing.
        Automatically detects file types (PDF, DOCX, PPTX, HTML, TXT, IMAGE, etc.) and extracts text content.
        For large files (configurable threshold), returns first, middle, and last chunks as bytes with separators.
        For images and PDFs with instructions, uses LLM processing if the model supports vision/PDF input.

        Usage Examples:
            - Read text file: {"file_path": "config.txt"}
            - Read PDF: {"file_path": "report.pdf"}
            - Per-page PDF read for downstream searches: {"file_path": "report.pdf", "document_mode": "page"}
            - Read DOCX: {"file_path": "document.docx"}
            - Read image: {"file_path": "image.png"} (extracts text using LLM)
            - Read large file: {"file_path": "large_data.json"}
            - Force summary preview: {"file_path": "report.pdf", "mode": "summary", "max_preview_bytes": 800}
            - Always chunk: {"file_path": "server.log", "mode": "chunked", "chunk_size_override": 4000}
            - Read image with instructions: {"file_path": "image.png", "instructions": "Describe the image in detail"}

        Parameters:
            - file_path: Path of the file to read
            - instructions: Optional instructions for LLM processing of images.
            - mode: "auto" (default), "full", "chunked", or "summary"
            - chunk_size_override: Optional override for chunk sizes in bytes/chars
            - max_preview_bytes: Optional cap for summary previews
            - document_mode: "file" (default) or "page" for per-page PDF extraction

        Notes:
            - Whenever text is extracted from non-text sources (PDF, PPTX, spreadsheets, etc.), it is cached as
              "<original_path>.extracted.txt" inside the same file store so FileSearchTool can reuse it without
              re-running converters.
    """
    llm: BaseLLM = Field(..., description="LLM that will be used to process files.")
    file_store: FileStore = Field(..., description="File storage to read from.")
    max_size: int = Field(default=10000, description="Maximum size in bytes before chunking (default: 10000)")
    chunk_size: int = Field(default=1000, description="Size of each chunk in bytes (default: 1000)")
    converter_mapping: dict[FileType, Node] | None = None
    spreadsheet_preview_rows: int = Field(
        default=5, description="Maximum number of rows to show per sheet when previewing spreadsheets."
    )
    spreadsheet_preview_max_chars: int = Field(
        default=8000, description="Maximum characters to emit per sheet preview to avoid massive outputs."
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)
    input_schema: ClassVar[type[FileReadInputSchema]] = FileReadInputSchema
    _connection_manager: ConnectionManager | None = PrivateAttr(default=None)
    _page_converter_cache: dict[FileType, Node] = PrivateAttr(default_factory=dict)

    def init_components(self, connection_manager: ConnectionManager | None = None):
        """
        Initialize the components of the FileReadTool.

        Args:
            connection_manager (ConnectionManager, optional): The connection manager to use.
                Defaults to a new ConnectionManager instance.
        """
        connection_manager = connection_manager or ConnectionManager()
        self._connection_manager = connection_manager
        super().init_components(connection_manager)

        self._setup_converters(connection_manager)

    def _setup_converters(self, connection_manager: ConnectionManager | None = None):
        """Setup internal converter components."""
        connection_manager = connection_manager or ConnectionManager()

        if not self.converter_mapping:
            self.converter_mapping = {}
            for file_type, converter_class in DEFAULT_FILE_TYPE_TO_CONVERTER_CLASS_MAP.items():
                if file_type == FileType.IMAGE and converter_class == LLMImageConverter:
                    self.converter_mapping[file_type] = converter_class(llm=self.llm)
                else:
                    self.converter_mapping[file_type] = converter_class()

        initialized_converters = set()
        for converter in self.converter_mapping.values():
            if id(converter) not in initialized_converters:
                if converter.is_postponed_component_init:
                    converter.init_components(connection_manager)
                initialized_converters.add(id(converter))
                logger.info(f"Initialized converter: {converter.name}")

    def _get_converter_for_type(self, file_type: FileType, document_mode: str) -> Node | None:
        """Fetch appropriate converter, supporting per-page PDF extraction."""
        if document_mode == "page" and file_type == FileType.PDF:
            cached = self._page_converter_cache.get(file_type)
            if not cached:
                page_converter = PyPDFConverter(document_creation_mode=PyPDFDocumentCreationMode.ONE_DOC_PER_PAGE)
                page_converter.init_components(self._connection_manager or ConnectionManager())
                self._page_converter_cache[file_type] = page_converter
                cached = page_converter
            return cached

        if not self.converter_mapping:
            return None
        return self.converter_mapping.get(file_type)

    def _detect_file_type(self, file: BytesIO, filename: str, config: RunnableConfig, **kwargs) -> FileType | None:
        """
        Detect the file type using custom detection function.

        Args:
            file: The file to analyze
            filename: The filename for file type detection
            config: Runtime configuration
            **kwargs: Additional arguments

        Returns:
            FileType: The detected file type, or None if detection fails
        """
        return detect_file_type(file, filename)

    def _process_file_with_converter(
        self,
        file: BytesIO,
        filename: str,
        detected_type: FileType,
        config: RunnableConfig,
        instructions: str | None = None,
        document_mode: str = "file",
        **kwargs,
    ) -> tuple[str | None, list[dict[str, Any]] | None]:
        """
        Process a file using the appropriate converter to extract text content.

        Args:
            file: The file to process
            filename: The filename
            detected_type: The detected file type
            config: Runtime configuration
            instructions: Custom instructions for image processing
            **kwargs: Additional arguments

        Returns:
            tuple[str | None, list[dict]]: Extracted text content and optional structured page data.
        """

        try:
            if detected_type == FileType.SPREADSHEET:
                spreadsheet_preview = self._render_spreadsheet_preview(file, filename)
                if spreadsheet_preview:
                    return spreadsheet_preview, None

            if detected_type in self.converter_mapping:

                converter = self._get_converter_for_type(detected_type, document_mode)
                if not converter:
                    logger.warning(f"No converter available for file type: {detected_type}")
                    return None, None

                if detected_type == FileType.IMAGE and instructions:
                    converter = LLMImageConverter(llm=self.llm, extraction_instruction=instructions)
                    converter_name = f"{converter.name} (with custom instructions)"
                else:
                    converter_name = converter.name

                file.seek(0)
                if not hasattr(file, "name"):
                    file.name = filename

                converter_input = {"files": [file]}
                result = converter.run(
                    input_data=converter_input,
                    config=config,
                    **(kwargs | {"parent_run_id": kwargs.get("run_id"), "run_depends": []}),
                )

                if result.status == RunnableStatus.SUCCESS:
                    documents = result.output.get("documents", [])
                    if documents:
                        logger.info(f"Successfully extracted text using {converter_name}")
                        if document_mode == "page":
                            page_entries = []
                            segments = []
                            for idx, doc in enumerate(documents, start=1):
                                page_num = doc.metadata.get("page_number") if doc.metadata else None
                                page_num = page_num or idx
                                content = doc.content
                                page_entries.append(
                                    {
                                        "page": page_num,
                                        "content": content,
                                        "metadata": doc.metadata or {},
                                    }
                                )
                                segments.append(f"=== PAGE {page_num} ===\n{content}")
                            return "\n\n".join(segments), page_entries

                        text_content = "\n\n".join([doc.content for doc in documents if hasattr(doc, "content")])
                        return text_content, None
                    else:
                        logger.warning(f"No documents extracted by {converter_name}")
                else:
                    logger.warning(f"Converter {converter_name} failed: {result.error}")

            else:
                logger.warning(f"No converter available for file type: {detected_type}")

        except Exception as e:
            logger.warning(f"File processing failed with converter: {str(e)}")
        return None, None

    def _render_spreadsheet_preview(self, file: BytesIO, filename: str) -> str | None:
        """Return a lightweight textual preview for spreadsheets using pandas head() per sheet."""
        try:
            import pandas as pd
        except Exception as exc:  # noqa: BLE001 optional dependency
            logger.debug("pandas unavailable for spreadsheet preview: %s", exc)
            return None

        try:
            file.seek(0)
            buffer = BytesIO(file.read())
        except Exception as exc:
            logger.warning("Failed to buffer spreadsheet %s for preview: %s", filename, exc)
            return None

        try:
            buffer.seek(0)
            with pd.ExcelFile(buffer) as excel_file:

                sheet_dimensions = self._sheet_dimensions_from_workbook(excel_file)

                limit = self.spreadsheet_preview_rows
                preview_segments: list[str] = [
                    f"Spreadsheet preview for '{filename}' "
                    f"({len(excel_file.sheet_names)} sheet{'s' if len(excel_file.sheet_names) != 1 else ''})."
                ]

                pd_options = pd.option_context("display.width", 120, "display.max_colwidth", 200)
                with pd_options:
                    for sheet_name in excel_file.sheet_names:
                        try:
                            preview_frame = excel_file.parse(sheet_name=sheet_name, nrows=limit)
                        except Exception as exc:
                            logger.warning("Failed to read preview rows for sheet %s: %s", sheet_name, exc)
                            continue

                        total_rows, total_columns = sheet_dimensions.get(
                            sheet_name,
                            (len(preview_frame.index), len(preview_frame.columns)),
                        )

                        preview_segments.append(
                            f"=== Sheet '{sheet_name or '(Unnamed Sheet)'}' "
                            f"— Rows: {total_rows:,}, Columns: {total_columns:,} "
                            f"(showing up to {limit} row(s)) ==="
                        )

                        if preview_frame.empty:
                            preview_segments.append("[Sheet is empty]")
                            continue

                        markdown = preview_frame.to_markdown(index=False)
                        truncated = False
                        max_chars = self.spreadsheet_preview_max_chars
                        if max_chars and len(markdown) > max_chars:
                            markdown = f"{markdown[: max_chars - 3]}..."
                            truncated = True

                        preview_segments.append(markdown)
                        if truncated:
                            preview_segments.append(f"[Preview truncated to {max_chars} characters]")

                        if total_rows > limit:
                            preview_segments.append(f"... showing only the first {limit} row(s).")

                return "\n\n".join(preview_segments) if len(preview_segments) > 1 else None

        except Exception as exc:
            logger.warning("Failed to open spreadsheet %s for preview: %s", filename, exc)
            return None

    @staticmethod
    def _sheet_dimensions_from_workbook(excel_file: Any) -> dict[str, tuple[int, int]]:
        workbook = getattr(excel_file, "book", None)
        if workbook is None:
            return {}

        dimensions: dict[str, tuple[int, int]] = {}
        for sheet_name in getattr(workbook, "sheetnames", []):
            worksheet = workbook[sheet_name]

            max_row = getattr(worksheet, "max_row", 0) or 0
            max_column = getattr(worksheet, "max_column", 0) or 0

            try:
                header_row = next(worksheet.iter_rows(values_only=True, max_row=1), None)
            except Exception:
                header_row = None
            has_header = bool(header_row and any(value is not None for value in header_row))

            total_rows = max(max_row - (1 if has_header else 0), 0)
            dimensions[sheet_name] = (total_rows, max_column)

        return dimensions

    @property
    def to_dict_exclude_params(self):
        """
        Property to define which parameters should be excluded when converting the class instance to a dictionary.

        Returns:
            dict: A dictionary defining the parameters to exclude.
        """
        return super().to_dict_exclude_params | {
            "llm": True,
            "converter_mapping": True,
        }

    def to_dict(self, **kwargs) -> dict:
        """Converts the instance to a dictionary.

        Returns:
            dict: A dictionary representation of the instance.
        """
        data = super().to_dict(**kwargs)
        data["llm"] = self.llm.to_dict(**kwargs)
        if self.converter_mapping:
            data["converter_mapping"] = {
                file_type.value: converter.to_dict(**kwargs) for file_type, converter in self.converter_mapping.items()
            }
        return data

    def execute(
        self,
        input_data: FileReadInputSchema,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Executes the file read operation and returns the file content.
        For large files, returns first, middle, and last chunks instead of full content.
        Automatically detects file type and extracts text content when possible.
        """
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")

        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        mode = input_data.mode or "auto"
        chunk_size = input_data.chunk_size_override or self.chunk_size
        chunk_size = max(chunk_size, 1)
        preview_limit = input_data.max_preview_bytes or chunk_size
        preview_limit = max(preview_limit, 1)
        allow_cache = input_data.instructions is None and input_data.document_mode == "file"

        try:
            if not self.file_store.exists(input_data.file_path):
                raise ToolExecutionException(
                    f"File '{input_data.file_path}' not found",
                    recoverable=True,
                )

            content = self.file_store.retrieve(input_data.file_path)
            content_size = len(content)

            cached_text, cached_path = (None, None)
            if allow_cache:
                cached_text, cached_path = self._load_cached_text(input_data.file_path)

            if cached_text:
                self._log_text_preview(cached_text, "cached extracted text")
                processed = self._render_text_content(
                    text_content=cached_text,
                    mode=mode,
                    chunk_size=chunk_size,
                    preview_limit=preview_limit,
                    file_path=input_data.file_path,
                )
                processed = self._append_cache_hint(processed, cached_path, hint_enabled=False)
                return {"content": processed, "cached_text_path": cached_path}

            try:
                file_io = BytesIO(content)
                filename = os.path.basename(input_data.file_path)

                detected_type = self._detect_file_type(file_io, filename, config, **kwargs)

                if detected_type:
                    text_content, page_entries = self._process_file_with_converter(
                        file_io,
                        filename,
                        detected_type,
                        config,
                        input_data.instructions,
                        input_data.document_mode,
                        **kwargs,
                    )

                    if text_content:
                        logger.info(
                            f"Tool {self.name} - {self.id}: successfully processed file and extracted text content"
                        )
                        self._log_text_preview(text_content, "extracted text")

                        cached_path = None
                        hint_enabled = False
                        if allow_cache:
                            cached_path = self._persist_extracted_text(input_data.file_path, text_content)
                            hint_enabled = detected_type not in {FileType.TEXT, FileType.MARKDOWN}

                        processed = self._render_text_content(
                            text_content=text_content,
                            mode=mode,
                            chunk_size=chunk_size,
                            preview_limit=preview_limit,
                            file_path=input_data.file_path,
                        )
                        processed = self._append_cache_hint(processed, cached_path, hint_enabled)
                        result_payload = {"content": processed}
                        if page_entries:
                            result_payload["pages"] = page_entries
                        if cached_path:
                            result_payload["cached_text_path"] = cached_path
                        return result_payload
                    else:
                        logger.warning(
                            f"Tool {self.name} - {self.id}: no text content extracted from file,"
                            "falling back to raw content"
                        )
                else:
                    logger.warning(
                        f"Tool {self.name} - {self.id}: could not detect file type, falling back to raw content"
                    )

            except Exception as e:
                logger.warning(
                    f"Tool {self.name} - {self.id}: file processing failed: {str(e)}, falling back to raw content"
                )

            rendered_content = self._render_binary_content(
                content=content,
                content_size=content_size,
                mode=mode,
                chunk_size=chunk_size,
                preview_limit=preview_limit,
                file_path=input_data.file_path,
            )

            return {"content": rendered_content}

        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to read file. Error: {str(e)}")
            raise ToolExecutionException(
                f"Tool '{self.name}' failed to read file. Error: {str(e)}. "
                f"Please analyze the error and take appropriate action.",
                recoverable=True,
            )

    def _create_chunked_content(self, content: bytes, chunk_size: int, file_path: str) -> bytes:
        """
        Create chunked content showing first, middle, and last parts of a large file.

        Args:
            content: The file content as bytes
            chunk_size: Size of each chunk in bytes
            file_path: Path of the file being read

        Returns:
            Concatenated bytes containing first, middle, and last chunks
        """
        total_size = len(content)

        first_chunk = content[:chunk_size]

        middle_start = total_size // 2 - chunk_size // 2
        middle_chunk = content[middle_start : middle_start + chunk_size]

        last_chunk = content[-chunk_size:] if total_size > chunk_size else content

        separator = f"\n\n--- CHUNKED FILE: {file_path} ({total_size:,} bytes total) ---\n".encode()
        first_sep = f"\n--- FIRST {len(first_chunk):,} BYTES ---\n".encode()
        middle_sep = f"\n--- MIDDLE {len(middle_chunk):,} BYTES (from position {middle_start:,}) ---\n".encode()
        last_sep = f"\n--- LAST {len(last_chunk):,} BYTES ---\n".encode()

        chunked_bytes = (
            separator
            + first_sep
            + first_chunk
            + middle_sep
            + middle_chunk
            + last_sep
            + last_chunk
            + b"\n\n--- END OF CHUNKED FILE ---\n"
        )

        return chunked_bytes

    def _create_chunked_text_content(self, content: str, chunk_size: int, file_path: str) -> str:
        """
        Create chunked text content showing first, middle, and last parts of a large text.

        Args:
            content: The text content as string
            chunk_size: Size of each chunk in characters
            file_path: Path of the file being read

        Returns:
            str: Concatenated string containing first, middle, and last chunks
        """
        total_size = len(content)

        first_chunk = content[:chunk_size]

        middle_start = total_size // 2 - chunk_size // 2
        middle_chunk = content[middle_start : middle_start + chunk_size]

        last_chunk = content[-chunk_size:] if total_size > chunk_size else content

        separator = f"\n\n--- CHUNKED TEXT FILE: {file_path} ({total_size:,} characters total) ---\n"
        first_sep = f"\n--- FIRST {len(first_chunk):,} CHARACTERS ---\n"
        middle_sep = f"\n--- MIDDLE {len(middle_chunk):,} CHARACTERS (from position {middle_start:,}) ---\n"
        last_sep = f"\n--- LAST {len(last_chunk):,} CHARACTERS ---\n"

        chunked_text = (
            separator
            + first_sep
            + first_chunk
            + middle_sep
            + middle_chunk
            + last_sep
            + last_chunk
            + "\n\n--- END OF CHUNKED TEXT FILE ---\n"
        )

        return chunked_text

    def _render_text_content(
        self, text_content: str, mode: str, chunk_size: int, preview_limit: int, file_path: str
    ) -> str:
        """Render text output according to the requested mode."""
        match mode:
            case "full":
                return text_content
            case "chunked":
                return self._create_chunked_text_content(text_content, chunk_size, file_path)
            case "summary":
                return self._create_summary_text_content(text_content, preview_limit, file_path)
            case _:
                if len(text_content) > self.max_size:
                    return self._create_chunked_text_content(text_content, chunk_size, file_path)
                return text_content

    def _render_binary_content(
        self, content: bytes, content_size: int, mode: str, chunk_size: int, preview_limit: int, file_path: str
    ) -> bytes | str:
        """Render binary output according to the requested mode."""
        match mode:
            case "full":
                return content
            case "chunked":
                return self._create_chunked_content(content, chunk_size, file_path)
            case "summary":
                return self._create_summary_bytes_content(content, preview_limit, file_path)
            case _:
                if content_size <= self.max_size:
                    return content
                return self._create_chunked_content(content, chunk_size, file_path)

    def _create_summary_text_content(self, content: str, max_chars: int, file_path: str) -> str:
        """Return a short preview string for text documents."""
        preview = content[:max_chars]
        suffix = "..." if len(content) > len(preview) else ""
        return (
            f"Preview of {file_path} (showing {len(preview):,} of {len(content):,} characters):\n" f"{preview}{suffix}"
        )

    def _create_summary_bytes_content(self, content: bytes, max_bytes: int, file_path: str) -> str:
        """Return a short preview string for binary files."""
        preview = content[:max_bytes]
        try:
            preview_text = preview.decode("utf-8")
            descriptor = "text"
        except UnicodeDecodeError:
            preview_text = preview.hex()
            descriptor = "hex"
        suffix = "..." if len(content) > len(preview) else ""
        return (
            f"Preview of {file_path} ({descriptor}, showing {len(preview):,} of {len(content):,} bytes):\n"
            f"{preview_text}{suffix}"
        )

    def _persist_extracted_text(self, original_path: str, text_content: str) -> str | None:
        """Persist extracted text so future reads/searches can reuse it."""
        if not self.file_store:
            return None

        cache_path = self._derived_cache_path(original_path)
        try:
            self.file_store.store(
                cache_path,
                text_content.encode("utf-8"),
                content_type="text/plain",
                metadata={
                    "source": "file_read_tool",
                    "original_path": original_path,
                },
                overwrite=True,
            )
            logger.info(f"Tool {self.name} - {self.id}: cached extracted text at {cache_path}")
            return cache_path
        except Exception as exc:
            logger.warning(f"Tool {self.name} - {self.id}: failed to cache extracted text for {original_path}: {exc}")
            return None

    def _load_cached_text(self, original_path: str) -> tuple[str | None, str | None]:
        """Load cached extracted text if it exists."""
        if not self.file_store:
            return None, None

        cache_path = self._derived_cache_path(original_path)
        try:
            if not self.file_store.exists(cache_path):
                return None, None
            cached_bytes = self.file_store.retrieve(cache_path)
            return cached_bytes.decode("utf-8"), cache_path
        except Exception as exc:
            logger.warning(f"Tool {self.name} - {self.id}: failed to load cached text for {original_path}: {exc}")
            return None, None

    @staticmethod
    def _derived_cache_path(original_path: str) -> str:
        return f"{original_path}{EXTRACTED_TEXT_SUFFIX}"

    @staticmethod
    def _append_cache_hint(content: str, cache_path: str | None, hint_enabled: bool = True) -> str:
        if cache_path and hint_enabled:
            hint = (
                f"\n\n[Extracted text cached at '{cache_path}'. "
                "Use FileSearchTool to search this processed content without re-reading the original file.]"
            )
            return f"{content}{hint}"
        return content

    def _log_text_preview(self, text: str, context: str, limit: int = 200) -> None:
        """Emit a short preview of extracted text so logs show what was parsed."""
        if not text:
            return
        preview = text.strip().replace("\n", " ")
        preview = preview[:limit]
        suffix = "..." if len(text.strip()) > limit else ""
        logger.info(
            f"Tool {self.name} - {self.id}: {context} preview ({min(len(preview), limit)} chars) => {preview}{suffix}"
        )


class FileWriteTool(Node):
    """
    A tool for writing files to storage.

    This tool can be passed to Agents to write files
    to the configured storage backend.

    Attributes:
        group (Literal[NodeGroup.TOOLS]): The group to which this tool belongs.
        name (str): The name of the tool.
        description (str): A brief description of the tool.
        file_store (FileStore): File storage to write to.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "FileWriteTool"
    description: str = """Writes files to storage based on the provided file path and content.

    Usage Examples:
    - Write text: {"file_path": "readme.txt", "content": "Hello World"}
    - Write JSON: {"file_path": "config.json", "content": {"key": "value"}}
    - Overwrite file: {"file_path": "existing.txt", "content": "new content"}
    - Append: {"file_path": "log.txt", "content": "\\nline", "append": true}
    - Force JSON encoding: {"file_path": "data.json", "content": {...}, "content_format": "json"}"""

    file_store: FileStore = Field(..., description="File storage to write to.")

    model_config = ConfigDict(arbitrary_types_allowed=True)
    input_schema: ClassVar[type[FileWriteInputSchema]] = FileWriteInputSchema

    def execute(
        self,
        input_data: FileWriteInputSchema,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Executes the file write operation and returns the file information.
        """
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")

        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        try:
            payload, inferred_type = self._prepare_content_payload(input_data)
            content_type = input_data.content_type or inferred_type

            overwrite_flag = input_data.overwrite or input_data.append
            if input_data.append and self.file_store.exists(input_data.file_path):
                existing = self.file_store.retrieve(input_data.file_path)
                payload = existing + payload

            file_info = self.file_store.store(
                input_data.file_path,
                payload,
                content_type=content_type,
                metadata=input_data.metadata,
                overwrite=overwrite_flag,
            )

            file_buffer = BytesIO(payload)
            file_buffer.name = input_data.file_path
            file_buffer.description = (input_data.metadata or {}).get("description", "FileWriteTool output")
            file_buffer.content_type = content_type
            file_buffer.seek(0)

            message = f"File '{input_data.file_path}' written successfully"
            logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(file_info)[:200]}...")
            return {
                "content": message,
                "file_info": file_info.model_dump(),
                "files": [file_buffer],
            }

        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to write file. Error: {str(e)}")
            raise ToolExecutionException(
                f"Tool '{self.name}' failed to write file. Error: {str(e)}. "
                f"Please analyze the error and take appropriate action.",
                recoverable=True,
            )

    def _prepare_content_payload(self, input_data: FileWriteInputSchema) -> tuple[bytes, str]:
        """Serialize the incoming payload based on the requested format."""
        encoding = input_data.encoding or "utf-8"
        fmt = input_data.content_format
        value = input_data.content

        if fmt == "auto":
            if isinstance(value, bytes):
                fmt = "binary"
            elif isinstance(value, (dict, list)):
                fmt = "json"
            else:
                fmt = "text"

        if fmt == "json":
            if isinstance(value, str):
                try:
                    json.loads(value)
                    serialized = value
                except json.JSONDecodeError:
                    serialized = json.dumps(value)
            else:
                serialized = json.dumps(value)
            return serialized.encode(encoding), "application/json"

        if fmt == "binary":
            if isinstance(value, bytes):
                return value, "application/octet-stream"
            return str(value).encode(encoding), "application/octet-stream"

        if isinstance(value, bytes):
            payload = value
        else:
            payload = str(value).encode(encoding)
        return payload, "text/plain"


class FileSearchInputSchema(BaseModel):
    """Schema for file search operations."""

    query: str = Field(..., description="Substring or regex to search for.")
    file_path: str | list[str] = Field(
        default="",
        description="Single path, list of paths, or empty to scan all available files.",
    )
    recursive: bool = Field(default=True, description="Whether to recurse when scanning directories.")
    mode: Literal["substring", "regex"] = Field(
        default="substring", description="Search mode: plain substring or regular expression."
    )
    case_sensitive: bool = Field(default=False, description="Whether the search should be case sensitive.")
    max_matches_per_file: int = Field(default=5, description="Maximum matches returned per file.")
    max_files: int = Field(default=20, description="Maximum number of files to scan when file_path is empty.")
    context_chars: int = Field(default=120, description="Number of characters of context to include around matches.")
    max_file_bytes: int = Field(
        default=1_000_000, description="Maximum number of bytes to load from each file while searching."
    )


class FileSearchTool(Node):
    """
    A tool for searching across stored files.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "FileSearchTool"
    description: str = """
        Searches stored files for substrings or regular expressions and returns contextual matches.
        Usage Examples:
            - {"query": "TODO"} (searches all files, case-insensitive substring)
            - {"query": "class Agent", "file_path": "dynamiq/nodes/agents/base.py"}
            - {"query": "error.+timeout", "mode": "regex", "case_sensitive": true}
            - {"query": "select", "context_chars": 300, "max_matches_per_file": 10}
        Notes:
            - When the FileReadTool has already extracted text (e.g., from PDF/PPTX/XLSX/CSV), this tool automatically
              searches the cached "<original>.extracted.txt" instead of re-reading the binary source.
            - Start with concrete phrases (e.g., "Global Drug Facility", "KPI tree") and widen or switch to regex
              only if needed; large, unfocused queries slow the agent down.
            - When you need page-level attribution, read PDFs with {"document_mode": "page"} so matches reference
              the same per-page extracts.
            - `context_chars` controls how much surrounding text is returned; increase it instead of re-running the
              same search repeatedly.
    """

    file_store: FileStore = Field(..., description="File storage to search within.")
    model_config = ConfigDict(arbitrary_types_allowed=True)
    input_schema: ClassVar[type[FileSearchInputSchema]] = FileSearchInputSchema

    def execute(
        self,
        input_data: FileSearchInputSchema,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")

        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        try:
            files_to_scan = self._resolve_files(input_data)
            matches = []
            total_scanned = 0

            if input_data.mode == "regex":
                flags = 0 if input_data.case_sensitive else re.IGNORECASE
                try:
                    pattern = re.compile(input_data.query, flags)
                except re.error as e:
                    raise ToolExecutionException(
                        f"Invalid regular expression '{input_data.query}': {e}", recoverable=True
                    )
            else:
                pattern = None
                query = input_data.query if input_data.case_sensitive else input_data.query.lower()

            for file_path in files_to_scan:
                total_scanned += 1
                file_matches = self._search_file(
                    file_path=file_path,
                    query=query if pattern is None else None,
                    pattern=pattern,
                    case_sensitive=input_data.case_sensitive,
                    max_matches=input_data.max_matches_per_file,
                    context_chars=input_data.context_chars,
                    max_bytes=input_data.max_file_bytes,
                )
                if file_matches:
                    matches.extend(file_matches)

            result = {
                "content": {
                    "matches": matches,
                    "files_scanned": total_scanned,
                    "total_matches": len(matches),
                }
            }
            logger.info(
                f"Tool {self.name} - {self.id}: finished searching {total_scanned} files, "
                f"found {len(matches)} matches."
            )
            return result

        except ToolExecutionException:
            raise
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to search files. Error: {str(e)}")
            raise ToolExecutionException(
                f"Tool '{self.name}' failed during search. Error: {str(e)}. "
                f"Please adjust your query or file selection and try again.",
                recoverable=True,
            )

    def _resolve_files(self, input_data: FileSearchInputSchema) -> list[str]:
        """Determine which files should be searched."""
        if isinstance(input_data.file_path, list) and input_data.file_path:
            return input_data.file_path[: input_data.max_files]
        if isinstance(input_data.file_path, str) and input_data.file_path:
            return [input_data.file_path]

        files = self.file_store.list_files(recursive=input_data.recursive)
        return [file.path for file in files[: input_data.max_files]]

    def _search_file(
        self,
        file_path: str,
        query: str | None,
        pattern: re.Pattern | None,
        case_sensitive: bool,
        max_matches: int,
        context_chars: int,
        max_bytes: int,
    ) -> list[dict[str, Any]]:
        """Run the search inside a single file."""
        text, source_path = self._load_search_text(file_path, max_bytes)
        if not text:
            return []

        matches: list[dict[str, Any]] = []

        if pattern:
            for match in pattern.finditer(text):
                matches.append(
                    self._build_match_entry(
                        file_path=file_path,
                        source_path=source_path,
                        text=text,
                        start=match.start(),
                        end=match.end(),
                        context_chars=context_chars,
                    )
                )
                if len(matches) >= max_matches:
                    break
        else:
            haystack = text if case_sensitive else text.lower()
            start_idx = 0
            query_len = len(query)
            while len(matches) < max_matches:
                idx = haystack.find(query, start_idx)
                if idx == -1:
                    break
                matches.append(
                    self._build_match_entry(
                        file_path=file_path,
                        source_path=source_path,
                        text=text,
                        start=idx,
                        end=idx + query_len,
                        context_chars=context_chars,
                    )
                )
                start_idx = idx + query_len

        return matches

    def _load_search_text(self, file_path: str, max_bytes: int) -> tuple[str | None, str | None]:
        """Load search text preferring cached extracted content when available."""
        if not self.file_store:
            return None, None

        candidates = [f"{file_path}{EXTRACTED_TEXT_SUFFIX}", file_path]
        for candidate in candidates:
            try:
                if not self.file_store.exists(candidate):
                    continue
                raw = self.file_store.retrieve(candidate)[:max_bytes]
                if not raw:
                    continue
                text = raw.decode("utf-8", errors="ignore")
                if text:
                    if candidate.endswith(EXTRACTED_TEXT_SUFFIX):
                        logger.info(
                            f"Tool {self.name} - {self.id}: using cached extracted text for search: {candidate}"
                        )
                    return text, candidate
            except Exception as exc:
                logger.warning(f"Tool {self.name} - {self.id}: failed to read {candidate} for search: {exc}")
        return None, None

    @staticmethod
    def _build_match_entry(
        file_path: str, source_path: str | None, text: str, start: int, end: int, context_chars: int
    ) -> dict[str, Any]:
        """Build a structured match entry with context and line number."""
        before = max(0, start - context_chars)
        after = min(len(text), end + context_chars)
        context = text[before:after]
        line = text.count("\n", 0, start) + 1
        return {
            "file": file_path,
            "source_path": source_path or file_path,
            "line": line,
            "match": text[start:end],
            "context": context.strip(),
        }


class FileListInputSchema(BaseModel):
    """Schema for file list input parameters."""

    file_path: str = Field(
        default="", description="Path of the file to list. Default is the root path. Keep empty to list all files."
    )
    recursive: bool = Field(default=True, description="Whether to list files recursively. Default is True.")


class FileListTool(Node):
    """
    A tool for listing files in storage.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "FileListTool"
    description: str = """Lists files in storage based on the provided file path."""

    file_store: FileStore = Field(..., description="File storage to list from.")
    model_config = ConfigDict(arbitrary_types_allowed=True)
    input_schema: ClassVar[type[FileListInputSchema]] = FileListInputSchema

    def execute(
        self,
        input_data: FileListInputSchema,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> dict[str, Any]:

        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")

        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        try:
            files_list = self.file_store.list_files(directory=input_data.file_path, recursive=input_data.recursive)
            files_string = "Files currently available in the filesystem storage:\n"
            for file in files_list:
                files_string += f"File: {file.name} | Path: {file.path} | Size: {file.size} bytes\n"

            logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{files_string}")
            return {"content": files_string}

        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to list files. Error: {str(e)}")
            raise ToolExecutionException(
                f"Tool '{self.name}' failed to list files. Error: {str(e)}. "
                f"Please analyze the error and take appropriate action.",
                recoverable=True,
            )
