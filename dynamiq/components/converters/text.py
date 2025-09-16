import copy
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

from charset_normalizer import from_bytes

from dynamiq.components.converters.base import BaseConverter
from dynamiq.components.converters.utils import get_filename_for_bytesio
from dynamiq.types import Document, DocumentCreationMode
from dynamiq.utils.logger import logger


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

        metadata = copy.deepcopy(metadata)
        metadata["file_path"] = filepath

        docs = [Document(content=content, metadata=metadata)]
        return docs

    def _detect_encoding(self, data: bytes) -> str:
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
