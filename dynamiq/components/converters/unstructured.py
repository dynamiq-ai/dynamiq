import base64
import copy
import enum
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import IO, Any

from unstructured_client import UnstructuredClient
from unstructured_client.models import operations, shared

from dynamiq.components.converters.base import BaseConverter
from dynamiq.components.converters.utils import get_filename_for_bytesio
from dynamiq.connections import Unstructured as UnstructuredConnection
from dynamiq.types import Document, DocumentCreationMode
from dynamiq.utils.logger import logger
from dynamiq.utils.utils import generate_uuid


class ConvertStrategy(str, enum.Enum):
    AUTO = "auto"
    FAST = "fast"
    HI_RES = "hi_res"
    OCR_ONLY = "ocr_only"
    VLM = "vlm"


class UnstructuredElementTypes(str, enum.Enum):
    FORMULA = "Formula"
    FIGURE_CAPTION = "FigureCaption"
    NARRATIVE_TEXT = "NarrativeText"
    LIST_ITEM = "ListItem"
    TITLE = "Title"
    ADDRESS = "Address"
    EMAIL_ADDRESS = "EmailAddress"
    IMAGE = "Image"
    PAGE_BREAK = "PageBreak"
    TABLE = "Table"
    HEADER = "Header"
    FOOTER = "Footer"
    CODE_SNIPPET = "CodeSnippet"
    PAGE_NUMBER = "PageNumber"
    UNCATEGORIZED_TEXT = "UncategorizedText"


def partition_via_api(
    filename: str | None = None,
    file: IO[bytes] | None = None,
    file_filename: str | None = None,
    api_url: str = "https://api.unstructured.io/",
    api_key: str = "",
    metadata_filename: str | None = None,
    extract_image_block_types: list[UnstructuredElementTypes] | None = None,
    **request_kwargs,
) -> list[dict]:
    """Partitions a document using the Unstructured REST API. This is equivalent to
    running the document through partition.

    See https://api.unstructured.io/general/docs for the hosted API documentation or
    https://github.com/Unstructured-IO/unstructured-api for instructions on how to run
    the API locally as a container.

    Parameters
    ----------
    filename
        A string defining the target filename path.
    content_type
        A string defining the file content in MIME type
    file
        A file-like object using "rb" mode --> open(filename, "rb").
    metadata_filename
        When file is not None, the filename (string) to store in element metadata. E.g. "foo.txt"
    api_url
        The URL for the Unstructured API. Defaults to the hosted Unstructured API.
    api_key
        The API key to pass to the Unstructured API.
    extract_image_block_types
        List of element types to extract as Base64-encoded representations.
        Common types include "Image", "Table". Element type names are case-insensitive.
        When specified, elements of these types will include an "image_base64" field in their metadata.
    request_kwargs
        Additional parameters to pass to the data field of the request to the Unstructured API.
    """

    if metadata_filename and file_filename:
        raise ValueError(
            "Only one of metadata_filename and file_filename is specified. "
            "metadata_filename is preferred. file_filename is marked for deprecation.",
        )

    if file_filename is not None:
        metadata_filename = file_filename
        logger.warning(
            "The file_filename kwarg will be deprecated in a future version of unstructured. "
            "Please use metadata_filename instead.",
        )

    base_url = api_url[:-19] if "/general/v0/general" in api_url else api_url
    client = UnstructuredClient(api_key_auth=api_key, server_url=base_url)

    files = None
    if filename is not None:
        with open(filename, "rb") as f:
            files = shared.Files(
                content=f.read(),
                file_name=filename,
            )

    elif file is not None:
        if metadata_filename is None:
            raise ValueError(
                "If file is specified in partition_via_api, "
                "metadata_filename must be specified as well.",
            )
        file.seek(0)
        files = shared.Files(
            content=file.read(),
            file_name=metadata_filename,
        )

    req_kwargs = request_kwargs.copy()
    if extract_image_block_types is not None:
        req_kwargs["extract_image_block_types"] = [element.value for element in extract_image_block_types]

    req = operations.PartitionRequest(
        partition_parameters=shared.PartitionParameters(
            files=files,
            **req_kwargs,
        )
    )
    response = client.general.partition(request=req)

    if response.status_code == 200:
        return response.elements
    else:
        raise ValueError(
            f"Receive unexpected status code {response.status_code} from the API.",
        )


class UnstructuredFileConverter(BaseConverter):
    """
    A component for converting files to Documents using the Unstructured API (hosted or running locally).

    For the supported file types and the specific API parameters, see
    [Unstructured docs](https://unstructured-io.github.io/unstructured/api.html).

    Usage example:
    ```python
    from dynamiq.components.converters.unstructured.file_converter import UnstructuredFileConverter

    # make sure to either set the environment variable UNSTRUCTURED_API_KEY
    # or run the Unstructured API locally:
    # docker run -p 8000:8000 -d --rm --name unstructured-api quay.io/unstructured-io/unstructured-api:latest
    # --port 8000 --host 0.0.0.0

    converter = UnstructuredFileConverter()
    documents = converter.run(paths=["a/file/path.pdf", "a/directory/path"])["documents"]
    ```
    """

    connection: UnstructuredConnection = None
    document_creation_mode: DocumentCreationMode = DocumentCreationMode.ONE_DOC_PER_FILE
    separator: str = "\n\n"
    strategy: ConvertStrategy = ConvertStrategy.AUTO
    unstructured_kwargs: dict[str, Any] | None = None
    extract_image_block_types_enabled: bool = False
    extract_image_block_types: list[UnstructuredElementTypes] | None = None

    def __init__(self, *args, **kwargs):
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = UnstructuredConnection()
        super().__init__(**kwargs)
        """
        Initializes the object with the configuration for converting documents using
        the Unstructured API.

        Args:
        connection (UnstructuredConnection, optional): The connection to use for the Unstructured API.
            Defaults to None, which will initialize a new UnstructuredConnection.
        document_creation_mode (Literal["one-doc-per-file", "one-doc-per-page", "one-doc-per-element"], optional):
            Determines how to create Documents from the elements returned by Unstructured. Options are:
            - `"one-doc-per-file"`: Creates one Document per file.
                All elements are concatenated into one text field.
            - `"one-doc-per-page"`: Creates one Document per page.
                All elements on a page are concatenated into one text field.
            - `"one-doc-per-element"`: Creates one Document per element.
                Each element is converted to a separate Document.
            Defaults to `"one-doc-per-file"`.
        separator (str, optional): The separator to use between elements when concatenating them into one text field.
            Defaults to "\n\n".
        strategy (Literal["auto", "fast", "hi_res", "ocr_only"], optional): The strategy to use for document processing.
            Defaults to "auto".
        unstructured_kwargs (Optional[dict[str, Any]], optional): Additional parameters to pass to the Unstructured API.
            See [Unstructured API docs](https://unstructured-io.github.io/unstructured/apis/api_parameters.html)
                for available parameters.
            Defaults to None.
        extract_image_block_types_enabled (bool, optional): Whether to extract and embed images/tables in the result.
            When enabled, Base64-encoded images and tables will be decoded and included in the document content.
            Defaults to False.
        extract_image_block_types (Optional[list[UnstructuredElementTypes]], optional): List of element types to extract
            when extract_image_block_types_enabled is True.
            If None and extract_image_block_types_enabled is True,
            defaults to [UnstructuredElementTypes.IMAGE, UnstructuredElementTypes.TABLE].
            Defaults to None.
        progress_bar (bool, optional): Whether to show a progress bar during the conversion process.
            Defaults to True.

        Returns:
        None
        """

    def _process_file(self, file: Path | str | BytesIO, metadata: dict[str, Any]) -> list[Any]:
        """
        Process a single file and create documents.

        Args:
            file (Union[Path, str, BytesIO]): The file to process.
            metadata (Dict[str, Any]): Metadata to attach to the documents.

        Returns:
            List[Any]: A list of created documents.

        Raises:
            ValueError: If the file object doesn't have a name and its extension can't be guessed.
            TypeError: If the file argument is neither a Path, string, nor a BytesIO object.
        """
        if isinstance(file, (Path, str)):
            file_name = str(file)
            elements = self._partition_file_into_elements_by_filepath(file_name)
        elif isinstance(file, BytesIO):
            file_name = get_filename_for_bytesio(file)
            elements = self._partition_file_into_elements_by_file(file, file_name)
        else:
            raise TypeError("Expected a Path object, a string path, or a BytesIO object.")
        return self._create_documents(
            filepath=file_name,
            elements=elements,
            document_creation_mode=self.document_creation_mode,
            metadata=metadata,
        )

    def _partition_file_into_elements_by_filepath(self, filepath: Path | str) -> list[dict[str, Any]]:
        """
        Partition a file into elements using the Unstructured API.

        Args:
            filepath (Path | str): The path to the file to partition.

        Returns:
            List[Dict[str, Any]]: A list of elements extracted from the file.

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file is empty or cannot be processed
            Exception: Any other exception that occurs during processing
        """
        if isinstance(filepath, str):
            filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        if filepath.stat().st_size == 0:
            raise ValueError(f"Empty file cannot be processed: {filepath}")

        extract_types = self.extract_image_block_types if self.extract_image_block_types_enabled else None
        if not extract_types and self.extract_image_block_types_enabled:
            extract_types = [UnstructuredElementTypes.IMAGE, UnstructuredElementTypes.TABLE]

        kwargs = copy.deepcopy(self.unstructured_kwargs) if self.unstructured_kwargs else {}
        if "extract_image_block_types" not in kwargs:
            kwargs["extract_image_block_types"] = extract_types

        return partition_via_api(
            filename=str(filepath),
            api_url=self.connection.url,
            api_key=self.connection.api_key,
            strategy=self.strategy,
            **kwargs,
        )

    def _partition_file_into_elements_by_file(
        self,
        file: BytesIO,
        metadata_filename: str,
    ) -> list[dict[str, Any]]:
        """
        Partition a file into elements using the Unstructured API.

        Args:
            file (BytesIO): The file object to partition.
            metadata_filename (str): The filename to store in element metadata.

        Returns:
            List[Dict[str, Any]]: A list of elements extracted from the file.

        Raises:
            ValueError: If the file is empty or cannot be processed
            Exception: Any other exception that occurs during processing
        """
        extract_types = self.extract_image_block_types if self.extract_image_block_types_enabled else None
        if not extract_types and self.extract_image_block_types_enabled:
            extract_types = [UnstructuredElementTypes.IMAGE, UnstructuredElementTypes.TABLE]

        kwargs = copy.deepcopy(self.unstructured_kwargs) if self.unstructured_kwargs else {}
        if "extract_image_block_types" not in kwargs:
            kwargs["extract_image_block_types"] = extract_types

        return partition_via_api(
            filename=None,
            file=file,
            metadata_filename=metadata_filename,
            api_url=self.connection.url,
            api_key=self.connection.api_key,
            strategy=self.strategy,
            **kwargs,
        )

    def _collect_images_and_tables(self, elements: list[dict]) -> tuple[list[dict], list[dict]]:
        """
        Collect images and tables from elements for separate processing.

        Args:
            elements (list[dict]): List of element dictionaries from Unstructured API.

        Returns:
            tuple: (images, tables) where each is a list of dicts with element data and metadata.
        """
        images = []
        tables = []

        for idx, element in enumerate(elements):
            metadata = element.get("metadata", {})

            if "image_base64" in metadata:
                base64_data = metadata["image_base64"]
                try:
                    decode_chars = max(16, min(50, len(base64_data)))
                    decoded_sample = base64.b64decode(base64_data[:decode_chars])
                    if decoded_sample.startswith(b"\xff\xd8\xff"):
                        image_format = "jpeg"
                    elif decoded_sample.startswith(b"\x89PNG"):
                        image_format = "png"
                    elif decoded_sample.startswith(b"GIF8"):
                        image_format = "gif"
                    elif (
                        len(decoded_sample) >= 12
                        and decoded_sample.startswith(b"RIFF")
                        and decoded_sample[8:12] == b"WEBP"
                    ):
                        image_format = "webp"
                    else:
                        image_format = "png"
                except Exception:
                    image_format = "png"

                images.append(
                    {
                        "id": generate_uuid(),
                        "index": idx,
                        "format": image_format,
                        "base64_data": base64_data,
                        "text": str(element.get("text", "")),
                        "element_metadata": metadata,
                    }
                )

            elif "text_as_html" in metadata:
                tables.append(
                    {
                        "index": idx,
                        "html": metadata["text_as_html"],
                        "text": str(element.get("text", "")),
                        "element_metadata": metadata,
                    }
                )

        return images, tables

    def _process_element_with_placeholder(self, element: dict, element_index: int, images: list[dict]) -> str:
        """
        Process an element with placeholders for images/tables instead of embedded content.

        Args:
            element (dict): The element dictionary from Unstructured API.
            element_index (int): Index of this element in the original elements list.
            images (list[dict]): List of collected images.
            tables (list[dict]): List of collected tables.

        Returns:
            str: The processed text content with placeholders for images/tables.
        """
        text = str(element.get("text", ""))

        if "text_as_html" in element.get("metadata", {}):
            text = element.get("metadata", {}).get("text_as_html", "")
        elif "image_base64" in element.get("metadata", {}):
            # Find corresponding image entry
            image_entry = next((i for i in images if i["index"] == element_index), None)
            if image_entry:
                image_id = image_entry["id"]
                placeholder_text = image_entry["text"].strip()
                if placeholder_text:
                    text = f"[IMAGE:{image_id}:{placeholder_text}]"
                else:
                    text = f"[IMAGE:{image_id}]"

        return text

    def _create_documents(
        self,
        filepath: str,
        elements: list[dict],
        document_creation_mode: DocumentCreationMode,
        metadata: dict[str, Any],
        **kwargs,
    ) -> list[Document]:
        """
        Create Documents from the elements returned by Unstructured.
        """
        separator = self.separator
        docs = []

        images, tables = self._collect_images_and_tables(elements)

        if document_creation_mode == DocumentCreationMode.ONE_DOC_PER_FILE:
            element_texts = []
            for idx, el in enumerate(elements):
                if self.extract_image_block_types_enabled:
                    text = self._process_element_with_placeholder(el, idx, images)
                else:
                    text = str(el.get("text", ""))

                if el.get("category") == "Title":
                    element_texts.append("# " + text)
                else:
                    element_texts.append(text)

            text = separator.join(element_texts)
            doc_metadata = copy.deepcopy(metadata)
            doc_metadata["file_path"] = str(filepath)

            if self.extract_image_block_types_enabled:
                if images:
                    for image in images:
                        image.pop("element_metadata", None)
                    doc_metadata["images"] = images
                if tables:
                    for table in tables:
                        table.pop("element_metadata", None)
                    doc_metadata["tables"] = tables

            docs = [Document(content=text, metadata=doc_metadata)]

        elif document_creation_mode == DocumentCreationMode.ONE_DOC_PER_PAGE:
            texts_per_page: defaultdict[int, str] = defaultdict(str)
            meta_per_page: defaultdict[int, dict] = defaultdict(dict)
            images_per_page: defaultdict[int, list] = defaultdict(list)
            tables_per_page: defaultdict[int, list] = defaultdict(list)

            for idx, el in enumerate(elements):
                if self.extract_image_block_types_enabled:
                    text = self._process_element_with_placeholder(el, idx, images)
                else:
                    text = str(el.get("text", ""))

                doc_metadata = copy.deepcopy(metadata)
                doc_metadata["file_path"] = str(filepath)
                element_metadata = el.get("metadata")
                if element_metadata:
                    doc_metadata.update(element_metadata)
                page_number = int(doc_metadata.get("page_number", 1))

                texts_per_page[page_number] += text + separator
                meta_per_page[page_number].update(doc_metadata)

                if self.extract_image_block_types_enabled:
                    # Find images/tables for this element
                    element_images = [img for img in images if img["index"] == idx]
                    element_tables = [tbl for tbl in tables if tbl["index"] == idx]
                    images_per_page[page_number].extend(element_images)
                    tables_per_page[page_number].extend(element_tables)

            for page in texts_per_page.keys():
                page_metadata = meta_per_page[page]
                if images_per_page[page]:
                    for image in images_per_page[page]:
                        image.pop("element_metadata", None)
                    page_metadata["images"] = images_per_page[page]
                if tables_per_page[page]:
                    for table in tables_per_page[page]:
                        table.pop("element_metadata", None)
                    page_metadata["tables"] = tables_per_page[page]

                docs.append(Document(content=texts_per_page[page], metadata=page_metadata))

        elif document_creation_mode == DocumentCreationMode.ONE_DOC_PER_ELEMENT:
            for index, el in enumerate(elements):
                if self.extract_image_block_types_enabled:
                    text = self._process_element_with_placeholder(el, index, images)
                else:
                    text = str(el.get("text", ""))

                doc_metadata = copy.deepcopy(metadata)
                doc_metadata["file_path"] = str(filepath)
                doc_metadata["element_index"] = index
                element_metadata = el.get("metadata", {})
                element_metadata = dict(element_metadata)

                # Add images/tables for this specific element
                if self.extract_image_block_types_enabled:
                    element_images = [img for img in images if img["index"] == index]
                    element_tables = [tbl for tbl in tables if tbl["index"] == index]
                    if element_images:
                        image_ids = [img["id"] for img in element_images]
                        if len(image_ids) == 1:
                            element_metadata["image_id"] = image_ids[0]
                        else:
                            element_metadata["image_ids"] = image_ids

                        for image in element_images:
                            image.pop("element_metadata", None)
                        doc_metadata["images"] = element_images
                    if element_tables:
                        for table in element_tables:
                            table.pop("element_metadata", None)
                        doc_metadata["tables"] = element_tables

                doc_metadata.update(element_metadata)
                element_category = el.get("category")
                if element_category:
                    doc_metadata["category"] = element_category

                doc = Document(content=text, metadata=doc_metadata)
                docs.append(doc)
        return docs
