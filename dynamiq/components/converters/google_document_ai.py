import copy
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeAlias

from pydantic import ConfigDict, Field, StringConstraints
from pypdf import PdfReader, PdfWriter

from dynamiq.components.converters.base import BaseConverter
from dynamiq.components.converters.utils import get_filename_for_bytesio
from dynamiq.connections import GoogleDocumentAI as GoogleDocumentAIConnection
from dynamiq.types import Document, DocumentCreationMode
from dynamiq.utils.logger import logger

if TYPE_CHECKING:
    from google.cloud.documentai import Document as GoogleDocument
    from google.cloud.documentai import DocumentProcessorServiceClient as DocumentAIClient

# The file types Document AI accepts for online processing. Declared to the API verbatim, as
# `RawDocument.mime_type` is a plain string. See https://cloud.google.com/document-ai/docs/file-types
PDF_MIME_TYPE = "application/pdf"
MIME_TYPES_BY_EXTENSION = {
    ".pdf": PDF_MIME_TYPE,
    ".bmp": "image/bmp",
    ".gif": "image/gif",
    ".jpe": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".webp": "image/webp",
}
SUPPORTED_MIME_TYPES = frozenset(MIME_TYPES_BY_EXTENSION.values())

MAX_PAGES_PER_REQUEST_IMAGELESS = 30
MAX_PAGES_PER_REQUEST_WITH_IMAGES = 15

DocumentAICreationMode: TypeAlias = Literal[
    DocumentCreationMode.ONE_DOC_PER_FILE,
    DocumentCreationMode.ONE_DOC_PER_PAGE,
]

ProcessorId: TypeAlias = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class GoogleDocumentAIFileConverter(BaseConverter):
    """
    A component that extracts plain text from documents using Google Document AI.

    Only the text layer of the processed document is kept - tables, form fields, entities and
    layout blocks are deliberately ignored. Use a splitter downstream to chunk the result.

    PDFs longer than the online page limit are split locally with pypdf and processed in
    consecutive batches, whose texts are then concatenated. Images are sent as-is, being
    single-page by nature. Any other file type is rejected before the request is made.

    Attributes:
        connection (GoogleDocumentAIConnection): The connection to the Document AI API. A new
            connection is created if none is provided. Used even when `client` is supplied,
            because the processor resource name is built from its project id and location.
        client (Any | None): An already initialized Document AI client. Used instead of connecting
            through `connection`.
        processor_id (str): The id of the Document AI processor to run, e.g. '1e0795e517c827d1'.
        processor_version (str | None): An optional pinned processor version, e.g. 'pretrained-ocr-v2.0-2023-06-02'.
            Defaults to None, which runs the processor's default version.
        document_creation_mode (DocumentCreationMode): Determines how to create Documents from the
            processed result. Options are:
            - `ONE_DOC_PER_FILE`: Creates one Document per file, with all page texts concatenated.
            - `ONE_DOC_PER_PAGE`: Creates one Document per page.
            Defaults to `ONE_DOC_PER_FILE`.
        enable_native_pdf_parsing (bool | None): Whether to read embedded text from digital PDFs
            instead of OCR-ing their rendered pages. Faster and more accurate for born-digital
            PDFs. Defaults to None, which omits OCR-specific options so non-OCR processors remain
            supported.
        imageless_mode (bool): Whether to omit page images from the response. Halves the payload
            and doubles the online page limit to 30. Defaults to True.
        page_separator (str): The separator inserted between page texts in
            `ONE_DOC_PER_FILE` mode. Defaults to "\\n\\n".

    Usage example:
        ```python
        from dynamiq.components.converters.google_document_ai import GoogleDocumentAIFileConverter
        from dynamiq.connections import GoogleDocumentAI

        converter = GoogleDocumentAIFileConverter(
            connection=GoogleDocumentAI(project_id="my-project", location="eu"),
            processor_id="1e0795e517c827d1",
        )
        documents = converter.run(file_paths=["a/file/path.pdf"])["documents"]
        ```
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    connection: GoogleDocumentAIConnection = Field(
        default_factory=GoogleDocumentAIConnection,
        description="Connection to the Document AI API. Supplies the project id and location.",
    )
    client: Any | None = Field(
        default=None,
        exclude=True,
        description="An already initialized Document AI client. Built from the connection when absent.",
    )
    processor_id: ProcessorId = Field(description="The id of the Document AI processor to run.")
    processor_version: str | None = Field(
        default=None,
        description="Optional pinned processor version. Defaults to the processor's default version.",
    )
    document_creation_mode: DocumentAICreationMode = Field(
        default=DocumentCreationMode.ONE_DOC_PER_FILE,
        description="Create one document per file, with page texts joined, or one per page.",
    )
    enable_native_pdf_parsing: bool | None = Field(
        default=None,
        description=(
            "Read embedded text from born-digital PDFs instead of OCR-ing rendered pages. "
            "When unset, OCR-specific process options are omitted."
        ),
    )
    imageless_mode: bool = Field(
        default=True,
        description="Omit page images from the response, which also doubles the online page limit.",
    )
    page_separator: str = Field(
        default="\n\n",
        description="Separator inserted between page texts in one-doc-per-file mode.",
    )

    @property
    def documentai_client(self) -> "DocumentAIClient":
        """The Document AI client, initialized from the connection on first use.

        Returns:
            DocumentProcessorServiceClient: The client used to process documents.
        """
        if self.client is None:
            self.client = self.connection.connect()
        return self.client

    @property
    def max_pages_per_request(self) -> int:
        """The online page limit that applies to the configured mode.

        Returns:
            int: 30 in imageless mode, 15 otherwise.
        """
        return MAX_PAGES_PER_REQUEST_IMAGELESS if self.imageless_mode else MAX_PAGES_PER_REQUEST_WITH_IMAGES

    @property
    def processor_name(self) -> str:
        """The fully qualified Document AI processor resource name.

        Returns:
            str: The processor version path when a version is pinned, otherwise the processor path.
        """
        if self.processor_version:
            return self.documentai_client.processor_version_path(
                self.connection.project_id,
                self.connection.location,
                self.processor_id,
                self.processor_version,
            )
        return self.documentai_client.processor_path(
            self.connection.project_id,
            self.connection.location,
            self.processor_id,
        )

    def _process_file(self, file: Path | BytesIO | bytes, metadata: dict[str, Any]) -> list[Any]:
        """
        Process a single file and create documents.

        Args:
            file (Path | BytesIO | bytes): The file to process.
            metadata (dict[str, Any]): Metadata to attach to the documents.

        Returns:
            list[Any]: A list of created documents.

        Raises:
            ValueError: If the file is empty, if its type is one Document AI cannot process, or if
                a BytesIO object has neither a name nor a guessable extension.
            TypeError: If the file argument is not a Path, a BytesIO object or bytes.
        """
        if isinstance(file, bytes):
            file = BytesIO(file)

        if isinstance(file, Path):
            file_path = str(file)
            content = file.read_bytes()
        elif isinstance(file, BytesIO):
            file.seek(0)
            file_path = get_filename_for_bytesio(file)
            file.seek(0)
            content = file.read()
        else:
            raise TypeError("Expected a Path object or a BytesIO object.")

        if not content:
            raise ValueError(f"File {file_path} is empty.")

        page_texts = self._extract_page_texts(content=content, mime_type=self._resolve_mime_type(file_path))

        return self._create_documents(
            filepath=file_path,
            elements=page_texts,
            document_creation_mode=self.document_creation_mode,
            metadata=metadata,
        )

    @staticmethod
    def _resolve_mime_type(file_path: str) -> str:
        """
        Resolve the MIME type to declare to Document AI from a file name.

        Args:
            file_path (str): The path or name of the file being processed.

        Returns:
            str: The guessed type, or PDF when the name carries no usable extension.

        Raises:
            ValueError: If the file type is one Document AI cannot process.
        """
        extension = Path(file_path).suffix.lower()
        if not extension:
            return PDF_MIME_TYPE

        if mime_type := MIME_TYPES_BY_EXTENSION.get(extension):
            return mime_type

        raise ValueError(
            f"File {file_path} has the unsupported extension '{extension}'. "
            f"Document AI accepts: {', '.join(sorted(SUPPORTED_MIME_TYPES))}."
        )

    def _extract_page_texts(self, content: bytes, mime_type: str) -> list[str]:
        """
        Extract the text of every page, splitting oversized PDFs into several requests.

        Args:
            content (bytes): The raw file content.
            mime_type (str): The MIME type to declare to Document AI.

        Returns:
            list[str]: The extracted text, one entry per page, in page order.
        """
        page_texts = []
        for request_content in self._request_batches(content, mime_type):
            page_texts.extend(self._page_texts(self._process_bytes(content=request_content, mime_type=mime_type)))

        return page_texts

    def _request_batches(self, content: bytes, mime_type: str) -> Iterator[bytes]:
        """
        Split the content into per-request payloads, in page order.

        Only PDFs over the online page limit are split; everything else is yielded as-is, either
        because it is single-page by nature or because pypdf could not read it - in which case
        Document AI is left to report the real error.

        Args:
            content (bytes): The raw file content.
            mime_type (str): The MIME type to declare to Document AI.

        Yields:
            bytes: The raw content of each request, in page order.
        """
        if mime_type != PDF_MIME_TYPE:
            yield content
            return

        pages_per_batch = self.max_pages_per_request
        try:
            reader = PdfReader(BytesIO(content))
            page_count = len(reader.pages)
        except Exception as e:
            logger.warning(f"Could not read the PDF page count locally, sending the file unsplit. Error: {e}")
            yield content
            return

        if page_count <= pages_per_batch:
            yield content
            return

        logger.debug(
            f"PDF has {page_count} pages, exceeding the {pages_per_batch}-page online limit. "
            f"Splitting it into batches."
        )
        for start in range(0, page_count, pages_per_batch):
            writer = PdfWriter()
            for page in reader.pages[start : start + pages_per_batch]:
                writer.add_page(page)

            batch = BytesIO()
            writer.write(batch)
            yield batch.getvalue()

    def _process_bytes(self, content: bytes, mime_type: str) -> "GoogleDocument":
        """
        Send a single online processing request to Document AI.

        Args:
            content (bytes): The raw file content.
            mime_type (str): The MIME type to declare to Document AI.

        Returns:
            Document: The processed Google Document AI document.
        """
        from google.cloud import documentai

        process_options = documentai.ProcessOptions()
        if self.enable_native_pdf_parsing is not None:
            process_options.ocr_config = documentai.OcrConfig(enable_native_pdf_parsing=self.enable_native_pdf_parsing)

        request = documentai.ProcessRequest(
            name=self.processor_name,
            raw_document=documentai.RawDocument(content=content, mime_type=mime_type),
            process_options=process_options,
            imageless_mode=self.imageless_mode,
        )

        return self.documentai_client.process_document(request=request).document

    @staticmethod
    def _page_texts(document: "GoogleDocument") -> list[str]:
        """
        Slice the document text into per-page texts.

        Document AI returns one flat text for the whole document plus, for each page, the text
        segments that page covers. Slicing by those segments keeps page boundaries without
        re-joining individual tokens.

        Args:
            document (Document): The processed Google Document AI document.

        Returns:
            list[str]: The text of each page, in page order.
        """
        text = document.text

        page_texts = []
        for page in document.pages:
            segments = page.layout.text_anchor.text_segments
            page_texts.append("".join(text[int(segment.start_index) : int(segment.end_index)] for segment in segments))

        if not page_texts and text:
            page_texts.append(text)

        return page_texts

    def _create_documents(
        self,
        filepath: str,
        elements: list[str],
        document_creation_mode: DocumentCreationMode,
        metadata: dict[str, Any],
        **kwargs,
    ) -> list[Document]:
        """
        Create Documents from the per-page texts returned by Document AI.

        Args:
            filepath (str): The path of the processed file.
            elements (list[str]): The text of each page, in page order.
            document_creation_mode (DocumentCreationMode): How to group pages into Documents.
            metadata (dict[str, Any]): Metadata to attach to the documents.

        Returns:
            list[Document]: The created Documents.
        """
        document_metadata = copy.deepcopy(metadata)
        document_metadata["file_path"] = filepath
        document_metadata["document_ai_processor_id"] = self.processor_id
        document_metadata["document_ai_pages"] = len(elements)
        if self.processor_version:
            document_metadata["document_ai_processor_version"] = self.processor_version

        if document_creation_mode == DocumentCreationMode.ONE_DOC_PER_PAGE:
            documents = []
            for page_number, page_text in enumerate(elements, start=1):
                page_metadata = copy.deepcopy(document_metadata)
                page_metadata["page_number"] = page_number
                documents.append(Document(content=page_text, metadata=page_metadata))
            return documents

        return [Document(content=self.page_separator.join(elements), metadata=document_metadata)]
