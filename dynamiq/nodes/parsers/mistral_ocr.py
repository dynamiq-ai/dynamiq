import base64
import mimetypes
import os
from io import BytesIO
from typing import Any, ClassVar, Literal
from urllib.parse import urljoin

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.connections import MistralOCR as MistralOCRConnection
from dynamiq.nodes.agents.utils import is_image_file
from dynamiq.nodes.node import ConnectionNode, ErrorHandling, ensure_config
from dynamiq.nodes.types import NodeGroup
from dynamiq.runnables import RunnableConfig

SUPPORTED_MIME_TYPES = [
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/epub+zip",
    "application/docbook+xml",
    "application/rtf",
    "application/vnd.oasis.opendocument.text",
    "application/x-biblatex",
    "application/x-bibtex",
    "application/x-endnote+xml",
    "application/x-fictionbook+xml",
    "application/x-ipynb+json",
    "application/x-jats+xml",
    "application/x-latex",
    "application/x-opml+xml",
    "text/troff",
    "text/x-dokuwiki",
]

IMAGE_MIME_TYPES = [
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/bmp",
    "image/tiff",
]


class MistralOCRInputSchema(BaseModel):
    """Schema for MistralOCR input data."""

    file: str | BytesIO | bytes = Field(
        ...,
        description="Parameter to provide file for OCR. "
        "Can be a file path, document URL, base64-encoded string, BytesIO object, or bytes.",
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)


class MistralOCR(ConnectionNode):
    """
    A component for extracting text from documents and images using Mistral's OCR capabilities.

    Attributes:
        group (Literal[NodeGroup.PARSERS]): The group the node belongs to.
        name (str): The name of the node.
        description (str): Description of the node functionality.
        model (str): The model to use for OCR (default: "mistral-ocr-latest").
        include_image_base64 (bool): Whether to include the base64-encoded images in the response (default: False).
        file_expiry_hours (int): Number of hours the uploaded file URL will be valid for (default: 1 hour).
        pages (list[int] | str | None): The pages to process (default: None). Supports these formats:
            - "3" (single page as a string),
            - "0-2" (range of pages as a string),
            - [0, 3, 4] (list of pages as a list of integers).
        image_limit (int | None): The maximum number of images to include in the response (default: None).
        image_min_size (int | None): The minimum size of images to include in the response (default: None).
        connection (MistralOCRConnection): The connection to the Mistral OCR API.
        error_handling (ErrorHandling): Error handling configuration.
    """

    group: Literal[NodeGroup.PARSERS] = NodeGroup.PARSERS
    name: str = "Mistral OCR"
    description: str = "Node that extracts text from documents using Mistral OCR API"
    model: str = "mistral-ocr-latest"
    include_image_base64: bool = False
    file_expiry_hours: int = 1
    pages: list[int] | str | None = None
    image_limit: int | None = None
    image_min_size: int | None = None
    mime_type_request_timeout: int = 10
    connection: MistralOCRConnection = Field(default_factory=MistralOCRConnection)
    error_handling: ErrorHandling = Field(default_factory=lambda: ErrorHandling(timeout_seconds=600))

    model_config = ConfigDict(arbitrary_types_allowed=True)
    input_schema: ClassVar[type[MistralOCRInputSchema]] = MistralOCRInputSchema

    def __init__(self, **kwargs):
        """Initialize the MistralOCR Node.

        If neither client nor connection is provided in kwargs, a new Mistral connection is created.

        Args:
            **kwargs: Keyword arguments to initialize the node.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = MistralOCRConnection()
        super().__init__(**kwargs)

    def _is_supported_file_type(self, file: str | BytesIO) -> bool:
        """
        Check if the file has a supported MIME type for Mistral OCR.

        Args:
            file (Union[str, BytesIO]): The file to check.

        Returns:
            bool: True if the file has a supported MIME type, False otherwise.
        """
        if isinstance(file, str) and (file.startswith("http") or file.startswith("https")):
            return True

        if isinstance(file, BytesIO) and is_image_file(file):
            return True

        if isinstance(file, str) and os.path.isfile(file):
            with open(file, "rb") as f:
                file_bytes = f.read()
                file_io = BytesIO(file_bytes)

            if is_image_file(file_io):
                return True

            mime_type, _ = mimetypes.guess_type(file)
            if mime_type:
                if mime_type in SUPPORTED_MIME_TYPES:
                    return True
                if mime_type.startswith("image/"):
                    return True

        if isinstance(file, BytesIO) and not is_image_file(file):
            content = file.getvalue()
            if content.startswith(b"%PDF"):
                return True

            if hasattr(file, "name") and file.name:
                mime_type, _ = mimetypes.guess_type(file.name)
                if mime_type:
                    if mime_type in SUPPORTED_MIME_TYPES:
                        return True
                    if mime_type.startswith("image/"):
                        return True

        return False

    def execute(self, input_data: MistralOCRInputSchema, config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        """
        Extract text from a document using Mistral OCR API.

        Args:
            input_data (MistralOCRInputSchema): Input data containing the document to process.
            config (RunnableConfig, optional): Configuration for the runnable, including callbacks.
            **kwargs: Additional arguments passed to the execution context.

        Returns:
            dict[str, Any]: A dictionary containing the extracted text and other OCR data.
        """
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        file = input_data.file

        if isinstance(file, bytes):
            file = BytesIO(file)

        try:
            if not self._is_supported_file_type(file):
                raise ValueError(
                    "Unsupported file type. OCR currently supports these mime types: " + ", ".join(SUPPORTED_MIME_TYPES)
                )

            is_url = isinstance(file, str) and (file.startswith("http") or file.startswith("https"))

            if is_url:
                ocr_result = self._parse_from_url(file)
            elif isinstance(file, BytesIO) and is_image_file(file):
                ocr_result = self._parse_image(file)
            elif isinstance(file, str) and os.path.isfile(file):
                with open(file, "rb") as f:
                    file_bytes = f.read()
                    file_io = BytesIO(file_bytes)

                if is_image_file(file_io):
                    ocr_result = self._parse_image(file_io)
                else:
                    ocr_result = self._process_via_upload(file)
            else:
                ocr_result = self._process_via_upload(file)

            return {"text": self._extract_text_from_result(ocr_result)}
        except Exception as e:
            msg = f"Encountered an error while performing OCR. \nError details: {e}"
            raise ValueError(msg)

    def _replace_images_in_markdown(self, markdown_string: str, images_dict: dict[str, Any]) -> str:
        """
        Replace image placeholders in output Markdown with actual image content.

        Args:
            markdown_string (str): The markdown string containing image placeholders.
            images_dict (dict[str, Any]): A dictionary mapping image IDs to their base64-encoded content.

        Returns:
            str: The markdown string with images replaced.
        """
        for image_name, base64_image_str in images_dict.items():
            markdown_string = markdown_string.replace(
                f"![{image_name}]({image_name})", f"![{image_name}]({base64_image_str})"
            )
        return markdown_string

    def _process_via_upload(self, file: str | BytesIO) -> dict[str, Any]:
        """
        Process a document via the upload workflow.

        Args:
            file (Union[str, BytesIO]): The file to process.

        Returns:
            dict[str, Any]: The OCR results.
        """
        upload_result = self._upload_file(file)
        file_id = upload_result.get("id")

        signed_url = self._get_signed_url(file_id, self.file_expiry_hours)
        document_url = signed_url.get("url")

        if not document_url:
            raise ValueError("Failed to get signed URL for uploaded file")

        return self._parse_from_url(document_url)

    def _upload_file(self, file: str | BytesIO) -> dict[str, Any]:
        """
        Upload a file to Mistral API.

        Args:
            file (Union[str, BytesIO]): The file to upload.

        Returns:
            dict[str, Any]: The upload response containing file ID.
        """
        files = None
        file_name = None

        if isinstance(file, BytesIO):
            file_content = file.getvalue()
            file_name = getattr(file, "name", "uploaded_file.pdf")
        elif isinstance(file, str):
            if os.path.isfile(file):
                with open(file, "rb") as f:
                    file_content = f.read()
                file_name = os.path.basename(file)
            else:
                raise ValueError(f"File not found: {file}")

        files = {"file": (file_name, file_content)}
        data = {"purpose": "ocr"}
        headers = {"Authorization": f"Bearer {self.connection.api_key}"}

        upload_url = urljoin(self.connection.base_url, "v1/files")
        response = self.client.request(
            method="POST",
            url=upload_url,
            headers=headers,
            files=files,
            data=data,
        )

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 401:
            raise ValueError("Invalid API key")
        else:
            raise ValueError(f"Failed to upload file: {response.status_code} {response.text}")

    def _get_mime_type_from_header(self, document_url: str) -> str:
        """
        Get the MIME type from the header of the document using its URL.

        Args:
            document_url (str): The URL of the document.

        Returns:
            str: The MIME type of the document obtained via HTTP HEAD request.

        Raises:
            ValueError: If the Content-Type header is missing or empty.
        """
        import requests

        head_response = requests.head(
            document_url,
            timeout=self.mime_type_request_timeout,
        )
        mime_type = head_response.headers.get("Content-Type", "")

        return mime_type

    def _get_signed_url(self, file_id: str, expiry_hours: int = 1) -> dict[str, Any]:
        """
        Get a signed URL for an uploaded file.

        Args:
            file_id (str): The ID of the uploaded file.
            expiry_hours (int): Number of hours the URL will be valid.

        Returns:
            dict[str, Any]: The response containing the signed URL.
        """
        url = urljoin(self.connection.base_url, f"v1/files/{file_id}/url")
        params = {"expiry": expiry_hours}
        headers = {"Authorization": f"Bearer {self.connection.api_key}", "Accept": "application/json"}

        response = self.client.request(
            method="GET",
            url=url,
            headers=headers,
            params=params,
        )

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 401:
            raise ValueError("Invalid API key")
        else:
            raise ValueError(f"Failed to get signed URL: {response.status_code} {response.text}")

    def _parse_from_url(
        self,
        document_url: str,
    ) -> dict[str, Any]:
        """
        Parse a document from a URL using Mistral OCR.

        Args:
            document_url (str): The URL of the document to process.

        Returns:
            dict[str, Any]: The OCR results containing extracted text and other data.
        """
        payload = {
            "model": self.model,
        }

        if document_url.startswith("http") or document_url.startswith("https"):
            mime_type, _ = mimetypes.guess_type(document_url)

            if not mime_type:
                mime_type = self._get_mime_type_from_header(document_url)

            if mime_type and mime_type in IMAGE_MIME_TYPES:
                payload["document"] = {"type": "image_url", "image_url": document_url}
            else:
                payload["document"] = {"type": "document_url", "document_url": document_url}

        additional_params = {
            "include_image_base64": self.include_image_base64,
            "pages": self.pages,
            "image_limit": self.image_limit,
            "image_min_size": self.image_min_size,
        }

        for key, value in additional_params.items():
            if value is not None:
                payload[key] = value

        headers = {"Authorization": f"Bearer {self.connection.api_key}", "Content-Type": "application/json"}

        ocr_url = urljoin(self.connection.base_url, "v1/ocr")
        response = self.client.request(
            method="POST",
            url=ocr_url,
            headers=headers,
            json=payload,
        )

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 401:
            raise ValueError("Invalid API key")
        else:
            raise ValueError(f"OCR request failed: {response.status_code} {response.text}")

    def _parse_image(
        self,
        image_file: BytesIO,
    ) -> dict[str, Any]:
        """
        Parse an image by encoding it as base64 and sending it to Mistral OCR.

        Args:
            image_file (BytesIO): The image to process.

        Returns:
            dict[str, Any]: The OCR results containing extracted text and other data.
        """
        image_bytes = image_file.getvalue()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        payload = {
            "model": self.model,
            "document": {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"},
        }

        additional_params = {
            "include_image_base64": self.include_image_base64,
            "pages": self.pages,
            "image_limit": self.image_limit,
            "image_min_size": self.image_min_size,
        }

        for key, value in additional_params.items():
            if value is not None:
                payload[key] = value

        headers = {"Authorization": f"Bearer {self.connection.api_key}", "Content-Type": "application/json"}

        ocr_url = urljoin(self.connection.base_url, "v1/ocr")
        response = self.client.request(
            method="POST",
            url=ocr_url,
            headers=headers,
            json=payload,
        )

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 401:
            raise ValueError("Invalid API key")
        else:
            raise ValueError(f"OCR request failed: {response.status_code} {response.text}")

    def _extract_text_from_result(self, result: dict[str, Any]) -> str:
        """
        Extract text from the OCR result.

        Args:
            result (dict[str, Any]): The OCR result.

        Returns:
            str: The extracted text.
        """
        markdown_pages = []
        if self.include_image_base64:
            for page in result["pages"]:
                image_data = {}
                for img in page["images"]:
                    image_data[img["id"]] = img["image_base64"]
                markdown_pages.append(self._replace_images_in_markdown(page["markdown"], image_data))
        else:
            for page in result["pages"]:
                if "markdown" in page:
                    markdown_pages.append(page["markdown"])

        return "\n\n".join(markdown_pages).strip()
