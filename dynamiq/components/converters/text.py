import codecs
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

from charset_normalizer import from_bytes

from dynamiq.components.converters.base import BaseConverter
from dynamiq.components.converters.utils import build_source_metadata, get_filename_for_bytesio
from dynamiq.types import Document, DocumentCreationMode
from dynamiq.utils.logger import logger

# Longest-prefix first: the 4-byte UTF-32 BOMs share their leading two bytes with the
# UTF-16 BOMs (``\xff\xfe`` / ``\xfe\xff``), so UTF-32 must be checked before UTF-16 or a
# UTF-32 file would be misread as UTF-16 with a two-byte "BOM" plus NUL padding.
_BOM_ENCODINGS: list[tuple[bytes, str]] = [
    (codecs.BOM_UTF32_LE, "utf-32"),
    (codecs.BOM_UTF32_BE, "utf-32"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
    (codecs.BOM_UTF8, "utf-8-sig"),
]


def detect_encoding(data: bytes) -> str:
    """
    Detect the encoding of the data using charset_normalizer.
    If detection fails, fallback to "utf-8".
    """
    try:
        result = from_bytes(data)
        best = result.best()

        if best and best.encoding:
            encoding = best.encoding

            try:
                data.decode(encoding)
                return encoding
            except UnicodeDecodeError:
                logger.debug(f"Detected encoding '{encoding}' failed to decode. Falling back...")

        else:
            logger.debug("Encoding detection returned None. Falling back...")

    except Exception as e:
        logger.debug(f"Encoding detection error: {e}. Falling back...")

    return "utf-8"


def decode_text_bytes(data: bytes) -> str:
    """
    Decode raw bytes to text, giving formats read raw (no converter pass) the same
    encoding handling ``.txt`` gets via ``TextFileConverter``.

    Attempts, cheapest and most conclusive first:
    1. A leading byte-order mark is an explicit, unambiguous encoding signal -- honour it
       before guessing anything.
    2. Strict UTF-8 covers the overwhelming majority of text files and costs nothing
       beyond the decode itself.
    3. ``charset_normalizer`` (via ``detect_encoding``) is comparatively expensive -- it
       scores multiple candidate encodings over the whole buffer -- so it only runs once
       the cheap paths have failed.
    4. UTF-8 with ``errors="replace"`` is the last resort so callers always get a string
       back, matching ``TextFileConverter``'s own fallback.
    """
    for bom, encoding in _BOM_ENCODINGS:
        if data.startswith(bom):
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                break

    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        pass

    encoding = detect_encoding(data)
    try:
        return data.decode(encoding)
    except UnicodeDecodeError:
        pass

    return data.decode("utf-8", errors="replace")


class TextFileConverter(BaseConverter):
    """
    A component for converting text files to Documents using the text file converter.

    Initializes the object with the configuration for converting documents using
    the text file converter.

    Args:
        document_creation_mode (Literal["one-doc-per-file"], optional):
            Determines how to create Documents from the text file content. Currently only supports:
            - `"one-doc-per-file"`: Creates one Document per file.
                All content is converted to markdown format.
            Defaults to `"one-doc-per-file"`.

    Usage example:
        ```python
        from dynamiq.components.converters.txt import TextFileConverter

        converter = TextFileConverter()
        documents = converter.run(paths=["a/file/path.txt", "a/directory/path"])["documents"]
        ```
    """

    document_creation_mode: Literal[DocumentCreationMode.ONE_DOC_PER_FILE] = DocumentCreationMode.ONE_DOC_PER_FILE

    def _process_file(self, file: Path | BytesIO, metadata: dict[str, Any]) -> list[Any]:
        """
        Process a file and return a list of Documents.

        Args:
            file: Path to a file or BytesIO object
            metadata: Metadata to attach to the documents

        Returns:
            List of Documents
        """

        if isinstance(file, BytesIO):
            filepath = get_filename_for_bytesio(file)
            file.seek(0)
            data = file.read()
        else:
            filepath = str(file)
            with open(file, "rb") as f:
                data = f.read()

        encoding = self._detect_encoding(data)
        content = data.decode(encoding, errors="replace")

        # Create documents from the text file content
        return self._create_documents(
            filepath=filepath,
            content=content,
            document_creation_mode=self.document_creation_mode,
            metadata=metadata,
        )

    def _create_documents(
        self,
        filepath: str,
        content: str,
        document_creation_mode: DocumentCreationMode,
        metadata: dict[str, Any],
        **kwargs,
    ) -> list[Document]:
        """
        Create Documents from the text content.
        """
        if document_creation_mode != DocumentCreationMode.ONE_DOC_PER_FILE:
            raise ValueError("TextFileConverter only supports one-doc-per-file mode")

        content = content.strip()
        if not content:
            raise ValueError(f"Text file '{filepath}' contains no extractable content.")

        metadata = build_source_metadata(metadata, filepath)

        docs = [Document(content=content, metadata=metadata)]
        return docs

    def _detect_encoding(self, data: bytes) -> str:
        """
        Detect the encoding of the data using charset_normalizer.
        If detection fails, fallback to "utf-8".

        Delegates to the module-level :func:`detect_encoding` so other callers (e.g.
        ``FileReadTool``'s raw plain-text path) get identical charset detection without
        reaching into this converter's internals.
        """
        return detect_encoding(data)
