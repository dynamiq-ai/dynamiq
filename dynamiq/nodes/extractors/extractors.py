import codecs
import enum
import re
import zipfile
from io import BytesIO
from typing import Any, ClassVar, Literal

import filetype
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


# Single source of truth: every category and the filename extensions that map to it. The reverse
# lookup below is derived from this, so a new format is added in exactly one place.
EXTENSION_MAP = {
    FileType.IMAGE: {
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

EXTENSION_TO_FILE_TYPE = {
    extension.lower(): file_type for file_type, extensions in EXTENSION_MAP.items() for extension in extensions
}

ZIP_DIRECTORY_MAP = {
    "word/": FileType.DOCUMENT,
    "ppt/": FileType.PRESENTATION,
    "xl/": FileType.SPREADSHEET,
}
ZIP_MIMETYPE_MAP = {
    "application/epub+zip": FileType.EBOOK,
    "application/vnd.oasis.opendocument.text": FileType.DOCUMENT,
    "application/vnd.oasis.opendocument.spreadsheet": FileType.SPREADSHEET,
    "application/vnd.oasis.opendocument.presentation": FileType.PRESENTATION,
}

# Leading bytes read for content detection.
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

    @staticmethod
    def _guess_type_from_zip(file: BytesIO | bytes) -> "FileType | None":
        """
        Resolve a ZIP-container format (OOXML/ODF/epub) by inspecting the archive's internal layout.

        Args:
            file (BytesIO | bytes): The file content, known to start with the ZIP signature.

        Returns:
            FileType | None: The resolved document category, ``FileType.ARCHIVE`` for a plain ZIP, or
                ``None`` if the bytes are not a readable ZIP archive.
        """
        if isinstance(file, BytesIO):
            source, position = file, file.tell()
        else:
            source, position = BytesIO(file), None

        try:
            source.seek(0)
            with zipfile.ZipFile(source) as archive:
                names = archive.namelist()
                if "mimetype" in names:
                    mimetype = archive.read("mimetype").decode("utf-8", errors="ignore").strip().lower()
                    file_type = ZIP_MIMETYPE_MAP.get(mimetype)
                    if file_type is not None:
                        return file_type
                for directory, file_type in ZIP_DIRECTORY_MAP.items():
                    if any(name.startswith(directory) for name in names):
                        return file_type
            return FileType.ARCHIVE
        except (zipfile.BadZipFile, OSError):
            return None
        finally:
            if position is not None:
                source.seek(position)

    @staticmethod
    def _guess_type_from_text(sample: bytes) -> "FileType | None":
        """
        Classify text-based content that carries no magic-byte signature.

        Args:
            sample (bytes): The leading bytes of the file.

        Returns:
            FileType | None: ``FileType.HTML`` or ``FileType.TEXT``, or ``None`` if the sample is not
                decodable as UTF-8 text.
        """
        try:
            text = codecs.getincrementaldecoder("utf-8")().decode(sample, final=False)
        except UnicodeDecodeError:
            return None

        head = text.lstrip().lower()
        if head.startswith(("<!doctype html", "<html")) or "<html" in head[:4096]:
            return FileType.HTML
        return FileType.TEXT

    @classmethod
    def _guess_type_from_content(cls, file: BytesIO | bytes | None) -> "FileType | None":
        """
        Detect the file category from raw content, covering ZIP-container, binary and text formats.

        Args:
            file (BytesIO | bytes | None): The file content to inspect.

        Returns:
            FileType | None: The detected category, or ``None`` when it cannot be determined.
        """
        header = cls._read_header(file, _CONTENT_SAMPLE_SIZE)
        if not header:
            return None

        # ZIP container: covers OOXML, ODF, epub and plain archives.
        if header.startswith(b"PK"):
            zip_type = cls._guess_type_from_zip(file)
            if zip_type is not None:
                return zip_type

        # Other binary formats with a stable magic-byte signature.
        extension = filetype.guess_extension(header)
        if extension:
            file_type = EXTENSION_TO_FILE_TYPE.get(extension.lower())
            if file_type is not None:
                return file_type

        # Text-based formats have no signature; fall back to content inspection.
        return cls._guess_type_from_text(header)

    def execute(
        self, input_data: FileTypeExtractorInputSchema, config: RunnableConfig = None, **kwargs
    ) -> dict[str, Any]:
        """
        Determines the category of a file from its filename extension when available, otherwise from
        its content.
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
