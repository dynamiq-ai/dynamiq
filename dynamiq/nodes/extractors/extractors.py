import enum
import re
from io import BytesIO
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.node import ensure_config
from dynamiq.runnables import RunnableConfig


class ByRegexExtractorInputSchema(BaseModel):
    value: str = Field(..., description="Parameter to provide value for extraction")
    pattern: str = Field(None, description="Parameter to specify regular expression pattern")


class ByRegexExtractor(Node):
    group: Literal[NodeGroup.EXTRACTORS] = NodeGroup.EXTRACTORS
    name: str = "by-regex-extractor"
    description: str = "Node that extracts data using regular expressions."
    pattern: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_schema: ClassVar[type[ByRegexExtractorInputSchema]] = ByRegexExtractorInputSchema

    def execute(
        self, input_data: ByRegexExtractorInputSchema, config: RunnableConfig = None, **kwargs
    ) -> dict[str, Any]:
        """
        Extract substrings or key-value pairs from text using regular expressions.

        Args:
            input_data (ByRegexExtractorInputSchema): input data for the tool, which includes value to transform and
                pattern to use.
            config (RunnableConfig, optional): Configuration for the runnable, including callbacks.
            **kwargs: Additional arguments passed to the execution context.

        Returns:
            dict[str, Any]: A dictionary containing list of matches.
        """
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        value = input_data.value
        pattern = input_data.pattern or self.pattern
        try:
            if pattern is None:
                raise ValueError("Pattern cannot be None")
            matches = re.finditer(pattern, value)

            result = [match.group() for match in matches]
            return {"matches": result}
        except Exception as e:
            raise ValueError(f"Encountered an error while performing extraction. \nError details: {e}")


class ByIndexExtractorInputSchema(BaseModel):
    input: list[Any] = Field(..., description="Parameter to provide list.")
    index: int | None = Field(None, description="Index of the first element in the list.")


class ByIndexExtractor(Node):
    group: Literal[NodeGroup.EXTRACTORS] = NodeGroup.EXTRACTORS
    name: str = "by-index-extractor"
    description: str = "Node that returns the element located at the given index in the list."
    index: int = 0

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_schema: ClassVar[type[ByIndexExtractorInputSchema]] = ByIndexExtractorInputSchema

    def execute(
        self, input_data: ByIndexExtractorInputSchema, config: RunnableConfig = None, **kwargs
    ) -> dict[str, Any]:
        """
        Returns the element at the specified index in the list.

        Args:
            input_data (ByIndexExtractorInputSchema): input data for the tool, which includes list for extraction.
            config (RunnableConfig, optional): Configuration for the runnable, including callbacks.
            **kwargs: Additional arguments passed to the execution context.

        Returns:
            dict[str, Any]: A dictionary containing element.
        """
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        list_object = input_data.input
        index = self.index if input_data.index is None else input_data.index
        if not list_object:
            raise ValueError("List has no elements inside.")
        elif 0 <= index < len(list_object):
            return {"output": list_object[index]}
        else:
            raise ValueError("Index out of range.")


class FileType(str, enum.Enum):
    IMAGE = "image"
    DOCUMENT = "document"
    PDF = "pdf"
    SPREADSHEET = "spreadsheet"
    PRESENTATION = "presentation"
    ARCHIVE = "archive"
    AUDIO = "audio"
    VIDEO = "video"
    FONT = "font"
    EXECUTABLE = "executable"
    DATABASE = "database"
    EBOOK = "ebook"
    HTML = "html"
    TEXT = "text"
    MARKDOWN = "markdown"


EXTENSION_MAP = {
    FileType.IMAGE: {
        "png",
        "jpg",
        "jpeg",
        "gif",
        "bmp",
        "webp",
        "tiff",
        "svg",
        "dwg",
        "xcf",
        "jpx",
        "apng",
        "cr2",
        "jxr",
        "psd",
        "ico",
        "heic",
        "avif",
    },
    FileType.DOCUMENT: {"doc", "docx", "odt", "rtf"},
    FileType.PDF: {"pdf"},
    FileType.SPREADSHEET: {"xls", "xlsx", "csv", "ods"},
    FileType.PRESENTATION: {"ppt", "pptx", "odp"},
    FileType.ARCHIVE: {
        "zip",
        "rar",
        "tar",
        "7z",
        "gz",
        "bz2",
        "xz",
        "lz",
        "lz4",
        "lzo",
        "zstd",
        "Z",
        "cab",
        "deb",
        "ar",
        "rpm",
        "br",
        "dcm",
        "ps",
        "crx",
    },
    FileType.AUDIO: {"mp3", "wav", "flac", "m4a", "aac", "ogg", "mid", "amr", "aiff"},
    FileType.VIDEO: {"mp4", "mkv", "avi", "mov", "wmv", "webm", "3gp", "m4v", "mpg", "flv"},
    FileType.FONT: {"woff", "woff2", "ttf", "otf"},
    FileType.EXECUTABLE: {"exe"},
    FileType.DATABASE: {"sqlite"},
    FileType.EBOOK: {"epub"},
    FileType.HTML: {"html"},
    FileType.TEXT: {"txt"},
    FileType.MARKDOWN: {"md"},
}


class FileTypeExtractorInputSchema(BaseModel):
    file: BytesIO | None = Field(None, description="Parameter to provide file")
    filename: str | None = Field("", description="Parameter to provide filename")
    model_config = ConfigDict(arbitrary_types_allowed=True)


class FileTypeExtractor(Node):
    group: Literal[NodeGroup.EXTRACTORS] = NodeGroup.EXTRACTORS
    name: str = "file-type-extractor"
    description: str = "Node that extract file category based on file extension"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_schema: ClassVar[type[FileTypeExtractorInputSchema]] = FileTypeExtractorInputSchema

    def execute(
        self, input_data: FileTypeExtractorInputSchema, config: RunnableConfig = None, **kwargs
    ) -> dict[str, Any]:
        """
        Determines the category of a file based on its extension.
        Args:
            input_data (FileTypeExtractorInputSchema): input data for the tool, which includes either a filename or a
                BytesIO file object.
            config (RunnableConfig, optional): Configuration for the runnable, including callbacks.
            **kwargs: Additional arguments passed to the execution context.
        Returns:
            dict[str, Any]: A dictionary containing file category.
        """
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        file = input_data.file
        filename = input_data.filename
        try:
            filename = getattr(file, "name", filename) if file else filename
            if not filename:
                raise ValueError("Invalid filename provided.")
            file_ext = filename.split(".")[-1] if "." in filename else ""
            file_ext = file_ext.lower()

            result = None
            for category, extensions in EXTENSION_MAP.items():
                if file_ext in extensions:
                    result = category
                    break

            return {"type": result}
        except Exception as e:
            raise ValueError(f"Encountered an error while performing extension extraction. \nError details: {e}")
