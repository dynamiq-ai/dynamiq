import copy
import enum
import json
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import IO, Any

from unstructured_client import UnstructuredClient
from unstructured_client.models import shared

from dynamiq.components.converters.base import BaseConverter
from dynamiq.components.converters.utils import get_filename_for_bytesio
from dynamiq.connections import Unstructured as UnstructuredConnection
from dynamiq.types import Document, DocumentCreationMode
from dynamiq.utils.logger import logger


class ConvertStrategy(str, enum.Enum):
    AUTO = "auto"
    FAST = "fast"
    HI_RES = "hi_res"
    OCR_ONLY = "ocr_only"


def partition_via_api(
    filename: str | None = None,
    content_type: str | None = None,
    file: IO[bytes] | None = None,
    file_filename: str | None = None,
    api_url: str = "https://api.unstructured.io/",
    api_key: str = "",
    metadata_filename: str | None = None,
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
    request_kwargs
        Additional parameters to pass to the data field of the request to the Unstructured API.
        For example the `strategy` parameter.
    """

    if metadata_filename and file_filename:
        raise ValueError(
            "Only one of metadata_filename and file_filename is specified. "
            "metadata_filename is preferred. file_filename is marked for deprecation.",
        )

    if file_filename is not None:
        metadata_filename = file_filename
        logger.warn(
            "The file_filename kwarg will be deprecated in a future version of unstructured. "
            "Please use metadata_filename instead.",
        )

    # Note(austin) - the sdk takes the base url, but we have the full api_url
    # For consistency, just strip off the path when it's given
    base_url = api_url[:-19] if "/general/v0/general" in api_url else api_url
    sdk = UnstructuredClient(api_key_auth=api_key, server_url=base_url)

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
        files = shared.Files(
            content=file,
            file_name=metadata_filename,
        )

    req = shared.PartitionParameters(
        files=files,
        **request_kwargs,
    )
    response = sdk.general.partition(req)

    if response.status_code == 200:
        element_dict = json.loads(response.raw_response.text)
        return element_dict
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
        progress_bar (bool, optional): Whether to show a progress bar during the conversion process.
            Defaults to True.

        Returns:
        None
        """

    def _process_file(
        self, file: Path | BytesIO, metadata: dict[str, Any]
    ) -> list[Any]:
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
            file_name = str(file)
            elements = self._partition_file_into_elements_by_filepath(file_name)
        elif isinstance(file, BytesIO):
            file_name = get_filename_for_bytesio(file)
            elements = self._partition_file_into_elements_by_file(file, file_name)
        else:
            raise TypeError("Expected a Path object or a BytesIO object.")
        return self._create_documents(
            filepath=file_name,
            elements=elements,
            document_creation_mode=self.document_creation_mode,
            metadata=metadata,
        )

    def _partition_file_into_elements_by_filepath(
        self, filepath: Path
    ) -> list[dict[str, Any]]:
        """
        Partition a file into elements using the Unstructured API.

        Args:
            filepath (Path): The path to the file to partition.

        Returns:
            List[Dict[str, Any]]: A list of elements extracted from the file.
        """
        try:
            return partition_via_api(
                filename=str(filepath),
                api_url=self.connection.url,
                api_key=self.connection.api_key,
                strategy=self.strategy,
                **self.unstructured_kwargs or {},
            )
        except Exception as e:
            logger.warning(
                f"Unstructured could not process file {filepath}. Error: {e}"
            )
            return []

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
        """
        try:
            return partition_via_api(
                filename=None,
                file=file,
                metadata_filename=metadata_filename,
                api_url=self.connection.url,
                api_key=self.connection.api_key,
                strategy=self.strategy,
                **self.unstructured_kwargs or {},
            )
        except Exception as e:
            logger.warning(
                f"Unstructured could not process file {metadata_filename}. Error: {e}"
            )
            return []

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
        if document_creation_mode == DocumentCreationMode.ONE_DOC_PER_FILE:
            element_texts = []
            for el in elements:
                text = str(el.get("text", ""))
                if el.get("category") == "Title":
                    element_texts.append("# " + text)
                else:
                    element_texts.append(text)

            text = separator.join(element_texts)
            metadata = copy.deepcopy(metadata)
            metadata["file_path"] = str(filepath)
            docs = [Document(content=text, metadata=metadata)]

        elif document_creation_mode == DocumentCreationMode.ONE_DOC_PER_PAGE:
            texts_per_page: defaultdict[int, str] = defaultdict(str)
            meta_per_page: defaultdict[int, dict] = defaultdict(dict)
            for el in elements:
                text = str(el.get("text", ""))
                metadata = copy.deepcopy(metadata)
                metadata["file_path"] = str(filepath)
                element_medata = el.get("metadata")
                if element_medata:
                    metadata.update(element_medata)
                page_number = int(metadata.get("page_number", 1))

                texts_per_page[page_number] += text + separator
                meta_per_page[page_number].update(metadata)

            docs = [
                Document(content=texts_per_page[page], metadata=meta_per_page[page])
                for page in texts_per_page.keys()
            ]

        elif document_creation_mode == DocumentCreationMode.ONE_DOC_PER_ELEMENT:
            for index, el in enumerate(elements):
                text = str(el.get("text", ""))
                metadata = copy.deepcopy(metadata)
                metadata["file_path"] = str(filepath)
                metadata["element_index"] = index
                element_medata = el.get("metadata")
                if element_medata:
                    metadata.update(element_medata)
                element_category = el.get("category")
                if element_category:
                    metadata["category"] = element_category
                doc = Document(content=str(el), metadata=metadata)
                docs.append(doc)
        return docs
