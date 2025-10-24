import logging
import os
from io import BytesIO
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

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
from dynamiq.nodes.llms.base import BaseLLM
from dynamiq.nodes.node import ensure_config
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.storages.file.base import FileStore
from dynamiq.utils.file_types import EXTENSION_MAP, FileType

logger = logging.getLogger(__name__)


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


class FileWriteInputSchema(BaseModel):
    """Schema for file write input parameters."""

    file_path: str = Field(..., description="Path where the file should be written")
    content: bytes | str = Field(..., description="File content (string, bytes)")
    content_type: str | None = Field(default=None, description="MIME type (auto-detected if not provided)")
    metadata: str | None = Field(default=None, description="Additional metadata for the file")


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
            - Read DOCX: {"file_path": "document.docx"}
            - Read image: {"file_path": "image.png"} (extracts text using LLM)
            - Read large file: {"file_path": "large_data.json"}
            - Read image with instructions: {"file_path": "image.png", "instructions": "Describe the image in detail"}

        Parameters:
            - file_path: Path of the file to read
            - instructions: Optional instructions for LLM processing of images.
    """
    llm: BaseLLM = Field(..., description="LLM that will be used to process files.")
    file_store: FileStore = Field(..., description="File storage to read from.")
    max_size: int = Field(default=10000, description="Maximum size in bytes before chunking (default: 10000)")
    chunk_size: int = Field(default=1000, description="Size of each chunk in bytes (default: 1000)")
    converter_mapping: dict[FileType, Node] | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)
    input_schema: ClassVar[type[FileReadInputSchema]] = FileReadInputSchema

    def init_components(self, connection_manager: ConnectionManager | None = None):
        """
        Initialize the components of the FileReadTool.

        Args:
            connection_manager (ConnectionManager, optional): The connection manager to use.
                Defaults to a new ConnectionManager instance.
        """
        connection_manager = connection_manager or ConnectionManager()
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
        **kwargs,
    ) -> str | None:
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
            str | None: Extracted text content from the file, or None if not available
        """

        try:
            if detected_type in self.converter_mapping:

                if detected_type == FileType.IMAGE and instructions:
                    converter = LLMImageConverter(llm=self.llm, extraction_instruction=instructions)
                    converter_name = f"{converter.name} (with custom instructions)"
                else:
                    converter = self.converter_mapping[detected_type]
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
                        text_content = "\n\n".join([doc.content for doc in documents if hasattr(doc, "content")])
                        logger.info(f"Successfully extracted text using {converter_name}")
                        return text_content
                    else:
                        logger.warning(f"No documents extracted by {converter_name}")
                else:
                    logger.warning(f"Converter {converter_name} failed: {result.error}")

            else:
                logger.warning(f"No converter available for file type: {detected_type}")

        except Exception as e:
            logger.warning(f"File processing failed with converter: {str(e)}")

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

        try:
            if not self.file_store.exists(input_data.file_path):
                raise ToolExecutionException(
                    f"File '{input_data.file_path}' not found",
                    recoverable=True,
                )

            content = self.file_store.retrieve(input_data.file_path)
            content_size = len(content)

            try:
                file_io = BytesIO(content)
                filename = os.path.basename(input_data.file_path)

                detected_type = self._detect_file_type(file_io, filename, config, **kwargs)

                if detected_type:
                    text_content = self._process_file_with_converter(
                        file_io, filename, detected_type, config, input_data.instructions, **kwargs
                    )

                    if text_content:
                        logger.info(
                            f"Tool {self.name} - {self.id}: successfully processed file and extracted text content"
                        )

                        # If the extracted text is large, return chunked content
                        if len(text_content) > self.max_size:
                            logger.info(
                                f"Tool {self.name} - {self.id}: extracted text is large ({len(text_content)} chars),"
                                " returning chunks"
                            )
                            chunked_content = self._create_chunked_text_content(
                                text_content, self.chunk_size, input_data.file_path
                            )
                            return {"content": chunked_content}
                        else:
                            return {"content": text_content}
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

            # Fallback to raw content if processing fails or no text extracted
            # If file is small enough, return full content
            if content_size <= self.max_size:
                logger.info(f"Tool {self.name} - {self.id}: returning full content ({content_size} bytes)")
                return {"content": content}

            # For large files, return chunked content
            logger.info(f"Tool {self.name} - {self.id}: file is large ({content_size} bytes), returning chunks")

            chunked_content = self._create_chunked_content(content, self.chunk_size, input_data.file_path)

            return {"content": chunked_content}

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
    - Overwrite file: {"file_path": "existing.txt", "content": "new content"}"""

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
            content_str = input_data.content
            if input_data.content_type is None:
                content_type = "text/plain"
            else:
                content_type = input_data.content_type

            # Store file
            file_info = self.file_store.store(
                input_data.file_path,
                content_str,
                content_type=content_type,
            )

            logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(file_info)[:200]}...")
            return {"content": f"File '{input_data.file_path}' written successfully"}

        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to write file. Error: {str(e)}")
            raise ToolExecutionException(
                f"Tool '{self.name}' failed to write file. Error: {str(e)}. "
                f"Please analyze the error and take appropriate action.",
                recoverable=True,
            )


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
