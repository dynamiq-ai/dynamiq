import copy
import enum
import re
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any, Literal, Mapping

from pypdf import PdfReader

from dynamiq.components.converters.base import BaseConverter
from dynamiq.components.converters.utils import get_filename_for_bytesio
from dynamiq.types import Document, DocumentCreationMode
from dynamiq.utils.logger import logger


class ExtractionMode(str, enum.Enum):
    PLAIN = "plain"
    LAYOUT = "layout"


class PyPDFFileConverter(BaseConverter):
    document_creation_mode: Literal[DocumentCreationMode.ONE_DOC_PER_FILE, DocumentCreationMode.ONE_DOC_PER_PAGE] = (
        DocumentCreationMode.ONE_DOC_PER_FILE
    )
    extraction_mode: ExtractionMode = ExtractionMode.PLAIN

    """
    A component for converting files to Documents using the PyPDFReader (hosted or running locally).

          Initializes the object with the configuration for converting documents using
          the PyPDF.

          Args:
          document_creation_mode (Literal["one-doc-per-file", "one-doc-per-page", "one-doc-per-element"], optional):
              Determines how to create Documents from the elements returned by PdfReader. Options are:
              - `"one-doc-per-file"`: Creates one Document per file.
                  All elements are concatenated into one text field.
              - `"one-doc-per-page"`: Creates one Document per page.
                  All elements on a page are concatenated into one text field.
              Defaults to `"one-doc-per-file"`.
          extraction_mode(ExtractionMode): Type of text extraction format.

        Usage example:
            ```python
            from dynamiq.components.converters.pypdf.file_converter import PyPdfFileConverter

            converter = PyPdfFileConverter()
            documents = converter.run(paths=["a/file/path.pdf", "a/directory/path"])["documents"]
            ```
    """

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

    def _process_file(self, file: Path | BytesIO, metadata: dict[str, Any]) -> list[Any]:
        """
        Process a single file and create documents.

        Args:
            file (Union[Path, BytesIO]): The file to process.
            metadata (Dict[str, Any]): Metadata to attach to the documents.

        Returns:
            List[Any]: A list of created documents.

        Raises:
            ValueError: If the file object doesn't have a name and its extension can't be guessed.
            TypeError: If the file argument is neither a Path nor a BytesIO object.
        """
        if isinstance(file, Path):
            file_path = str(file)
            elements = self._partition_file_into_elements(file_path)
        elif isinstance(file, BytesIO):
            file_path = get_filename_for_bytesio(file)
            elements = self._partition_file_into_elements(file)
        else:
            raise TypeError("Expected a Path object or a BytesIO object.")
        return self._create_documents(
            filepath=file_path,
            elements=elements,
            document_creation_mode=self.document_creation_mode,
            metadata=metadata,
        )

    @staticmethod
    def _partition_file_into_elements(
        file: Path | BytesIO,
    ) -> PdfReader:
        """
        Partition a file into elements using the PyPDFReader.

        Args:
            filepath (Path): The path to the file to partition.

        Returns:
            PdfReader: PdfReader object with elements extracted from the file.
        """
        try:
            return PdfReader(file)
        except Exception as e:
            logger.warning(f"PyPDF converter could not process file {file.name or str(file)}. Error: {e}")
            return []

    def _create_documents(
        self,
        filepath: str,
        elements: PdfReader,
        document_creation_mode: DocumentCreationMode,
        metadata: dict[str, Any],
        **kwargs,
    ) -> list[Document]:
        """
        Create Documents from the elements returned by PyPDF.
        """
        document_metadata = copy.deepcopy(metadata)
        document_metadata["file_path"] = filepath
        document_metadata.update(self._normalize_pdf_metadata(elements.metadata))

        if document_creation_mode == DocumentCreationMode.ONE_DOC_PER_FILE:
            text = "".join(page.extract_text(extraction_mode=self.extraction_mode) for page in elements.pages)
            return [Document(content=text, metadata=document_metadata)]

        if document_creation_mode == DocumentCreationMode.ONE_DOC_PER_PAGE:
            texts_per_page: defaultdict[int, str] = defaultdict(str)
            meta_per_page: defaultdict[int, dict] = defaultdict(dict)
            for page_number, el in enumerate(elements.pages, start=1):
                text = str(el.extract_text(extraction_mode=self.extraction_mode))
                page_metadata = copy.deepcopy(document_metadata)
                page_metadata["page_number"] = page_number

                texts_per_page[page_number] += text
                meta_per_page[page_number].update(page_metadata)

            return [
                Document(content=texts_per_page[page], metadata=meta_per_page[page]) for page in texts_per_page.keys()
            ]

        return []

    @staticmethod
    def _normalize_pdf_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
        """Namespace PDF Info keys as safe, predictable document metadata keys."""
        normalized: dict[str, Any] = {}
        for raw_key, value in (metadata or {}).items():
            key = str(raw_key).lstrip("/")
            key = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", key)
            key = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key)
            key = re.sub(r"[^A-Za-z0-9]+", "_", key).strip("_").lower()
            base_key = f"pdf_{key or 'metadata'}"
            normalized_key = base_key
            suffix = 2
            while normalized_key in normalized:
                normalized_key = f"{base_key}_{suffix}"
                suffix += 1
            normalized[normalized_key] = value
        return normalized
