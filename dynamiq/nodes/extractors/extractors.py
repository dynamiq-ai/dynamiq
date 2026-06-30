import enum
import re
from io import BytesIO
from typing import Any, ClassVar, Literal

import magic
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


FILE_TYPE_DEFINITIONS: dict[FileType, dict[str, set[str]]] = {
    FileType.IMAGE: {
        "extensions": {
            "png",
            "jpg",
            "jpeg",
            "gif",
            "bmp",
            "webp",
            "tiff",
            "tif",
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
        "mimes": {
            "image/png",
            "image/jpeg",
            "image/gif",
            "image/bmp",
            "image/webp",
            "image/tiff",
            "image/svg+xml",
            "image/x-icon",
            "image/vnd.microsoft.icon",
            "image/heic",
            "image/heif",
            "image/avif",
            "image/vnd.adobe.photoshop",
            "image/x-canon-cr2",
            "image/vnd.dwg",
            "image/x-xcf",
        },
    },
    FileType.DOCUMENT: {
        "extensions": {"doc", "docx", "odt", "rtf"},
        "mimes": {
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.oasis.opendocument.text",
            "application/rtf",
            "text/rtf",
        },
    },
    FileType.PDF: {
        "extensions": {"pdf"},
        "mimes": {"application/pdf"},
    },
    FileType.SPREADSHEET: {
        "extensions": {"xls", "xlsx", "csv", "ods"},
        "mimes": {
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.oasis.opendocument.spreadsheet",
            "text/csv",
            "application/csv",
        },
    },
    FileType.PRESENTATION: {
        "extensions": {"ppt", "pptx", "odp"},
        "mimes": {
            "application/vnd.ms-powerpoint",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "application/vnd.oasis.opendocument.presentation",
        },
    },
    FileType.ARCHIVE: {
        "extensions": {
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
            "zst",
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
        "mimes": {
            "application/zip",
            "application/x-rar",
            "application/vnd.rar",
            "application/x-tar",
            "application/x-7z-compressed",
            "application/gzip",
            "application/x-bzip2",
            "application/x-xz",
            "application/x-lzip",
            "application/x-lz4",
            "application/zstd",
            "application/x-compress",
            "application/vnd.ms-cab-compressed",
            "application/x-deb",
            "application/x-archive",
            "application/x-rpm",
            "application/x-redhat-package-manager",
            "application/dicom",
        },
    },
    FileType.AUDIO: {
        "extensions": {"mp3", "wav", "flac", "m4a", "aac", "ogg", "mid", "amr", "aiff"},
        "mimes": {
            "audio/mpeg",
            "audio/wav",
            "audio/x-wav",
            "audio/flac",
            "audio/x-flac",
            "audio/mp4",
            "audio/aac",
            "audio/ogg",
            "audio/midi",
            "audio/x-midi",
            "audio/amr",
            "audio/aiff",
            "audio/x-aiff",
        },
    },
    FileType.VIDEO: {
        "extensions": {"mp4", "mkv", "avi", "mov", "wmv", "webm", "3gp", "m4v", "mpg", "flv"},
        "mimes": {
            "video/mp4",
            "video/x-matroska",
            "video/x-msvideo",
            "video/quicktime",
            "video/x-ms-wmv",
            "video/webm",
            "video/3gpp",
            "video/mpeg",
            "video/x-flv",
        },
    },
    FileType.FONT: {
        "extensions": {"woff", "woff2", "ttf", "otf"},
        "mimes": {
            "font/woff",
            "font/woff2",
            "font/ttf",
            "font/otf",
            "application/font-woff",
            "application/font-sfnt",
            "application/x-font-ttf",
            "application/vnd.ms-opentype",
        },
    },
    FileType.EXECUTABLE: {
        "extensions": {"exe"},
        "mimes": {
            "application/x-dosexec",
            "application/x-msdownload",
            "application/vnd.microsoft.portable-executable",
        },
    },
    FileType.DATABASE: {
        "extensions": {"sqlite"},
        "mimes": {"application/x-sqlite3", "application/vnd.sqlite3"},
    },
    FileType.EBOOK: {
        "extensions": {"epub"},
        "mimes": {"application/epub+zip"},
    },
    FileType.HTML: {
        "extensions": {"html"},
        "mimes": {"text/html"},
    },
    FileType.TEXT: {
        "extensions": {"txt"},
        "mimes": {"text/plain"},
    },
    FileType.MARKDOWN: {
        "extensions": {"md"},
        "mimes": {"text/markdown", "text/x-markdown"},
    },
}

EXTENSION_MAP = {file_type: definition["extensions"] for file_type, definition in FILE_TYPE_DEFINITIONS.items()}

EXTENSION_TO_FILE_TYPE = {
    extension.lower(): file_type
    for file_type, definition in FILE_TYPE_DEFINITIONS.items()
    for extension in definition["extensions"]
}
MIME_TO_FILE_TYPE = {
    mime.lower(): file_type for file_type, definition in FILE_TYPE_DEFINITIONS.items() for mime in definition["mimes"]
}

# Leading bytes handed to libmagic.
_CONTENT_SAMPLE_SIZE = 8192


class FileTypeExtractorInputSchema(BaseModel):
    file: BytesIO | bytes | None = Field(None, description="Parameter to provide file content as bytes or BytesIO.")
    filename: str | None = Field("", description="Parameter to provide filename")
    model_config = ConfigDict(arbitrary_types_allowed=True)


class FileTypeExtractor(Node):
    group: Literal[NodeGroup.EXTRACTORS] = NodeGroup.EXTRACTORS
    name: str = "file-type-extractor"
    description: str = "Node that extract file category based on file extension or content"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_schema: ClassVar[type[FileTypeExtractorInputSchema]] = FileTypeExtractorInputSchema

    @staticmethod
    def _read_header(file: BytesIO | bytes | None, size: int) -> bytes:
        """
        Read up to ``size`` leading bytes without consuming a ``BytesIO`` stream.
        Args:
            file (BytesIO | bytes | None): The file content to read.
            size (int): Maximum number of leading bytes to return.

        Returns:
            bytes: The leading bytes, or an empty ``bytes`` if nothing readable was provided.
        """
        if file is None:
            return b""

        if isinstance(file, bytes):
            return file[:size]

        if isinstance(file, BytesIO):
            position = file.tell()
            file.seek(0)
            try:
                return file.read(size)
            finally:
                file.seek(position)

        return b""

    @classmethod
    def _guess_type_from_content(cls, file: BytesIO | bytes | None) -> "FileType | None":
        """
        Detect the file category from raw content using libmagic, mapped through ``MIME_TO_FILE_TYPE``.

        Args:
            file (BytesIO | bytes | None): The file content to inspect.

        Returns:
            FileType | None: The detected category, or ``None`` when the content is empty or its MIME
                type is not mapped to a known category.
        """
        header = cls._read_header(file, _CONTENT_SAMPLE_SIZE)
        if not header:
            return None

        mime_type = magic.from_buffer(header, mime=True)
        if not mime_type:
            return None

        mime_type = mime_type.split(";")[0].strip().lower()
        return MIME_TO_FILE_TYPE.get(mime_type)

    def execute(
        self, input_data: FileTypeExtractorInputSchema, config: RunnableConfig = None, **kwargs
    ) -> dict[str, Any]:
        """
        Determines the category of a file from its filename extension when available, otherwise from
        its content via MIME detection.
        Args:
            input_data (FileTypeExtractorInputSchema): input data for the tool, which includes a filename
                and/or the file content as ``bytes`` or a ``BytesIO`` object.
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
            filename = (getattr(file, "name", None) or filename) if file is not None else filename

            result = None

            if filename and "." in filename:
                file_ext = filename.rsplit(".", 1)[-1].lower()
                result = EXTENSION_TO_FILE_TYPE.get(file_ext)

            if result is None:
                result = self._guess_type_from_content(file)

            return {"type": result}
        except Exception as e:
            raise ValueError(f"Encountered an error while performing extension extraction. \nError details: {e}")
