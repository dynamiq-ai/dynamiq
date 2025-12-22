import base64
import copy
import io
from typing import Any, Callable, ClassVar, Literal

import filetype
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes import ErrorHandling
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode, NodeGroup, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

from .generation import ImageResponseFormat, ImageSize, create_image_file, download_image_from_url


def prepare_single_image(img: bytes | io.BytesIO) -> io.BytesIO:
    """Prepare the image for the API call.

    Args:
        img: Image to prepare (bytes, BytesIO).

    Returns:
        BytesIO file-like object for API submission.
    """
    if isinstance(img, bytes):
        image_bytes = img
    elif isinstance(img, io.BytesIO):
        img.seek(0)
        image_bytes = img.read()
    else:
        raise ValueError(f"Unsupported image type: {type(img)}")

    try:
        img_obj = Image.open(io.BytesIO(image_bytes))
        img_obj.load()
    except Exception as e:
        raise ValueError("Invalid image data") from e

    original_format = img_obj.format

    if not original_format:
        kind = filetype.guess(image_bytes)
        if not kind:
            raise ValueError("Unable to detect image format")

        original_format = kind.extension.upper()
        original_format = "JPEG" if original_format == "JPG" else original_format

    if original_format in ("JPEG", "JPG"):
        if img_obj.mode not in ("RGB", "L"):
            img_obj = img_obj.convert("RGB")
    else:
        if img_obj.mode not in ("RGBA", "LA", "L"):
            img_obj = img_obj.convert("RGBA")

    output_bytes = io.BytesIO()
    img_obj.save(output_bytes, format=original_format)
    output_bytes.seek(0)
    return output_bytes


class ImageEditInputSchema(BaseModel):
    """Input schema for image editing."""

    prompt: str = Field(..., description="Text prompt describing the desired edits.")
    files: list[io.BytesIO | bytes] | io.BytesIO | bytes = Field(
        ...,
        description="The image(s) to edit. Can be a single image or a list of images.",
        json_schema_extra={"map_from_storage": True, "is_accessible_to_agent": False},
    )
    mask: io.BytesIO | bytes | None = Field(
        default=None,
        description="Optional mask image indicating areas to edit.",
    )
    n: int | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ImageEdit(ConnectionNode):
    """
    Node for editing images using AI models.

    Takes an existing image and a text prompt to generate edited versions.
    Optionally accepts a mask to specify which areas to modify.

    Attributes:
        FILE_PREFIX (str): Prefix for new file names. Default to "edited".
        name (str): The name of the node.
        model (str): The model to use for image editing (e.g., 'dall-e-2', 'gpt-image-1').
        connection (OpenAIConnection): The connection to the API.
        n (int): Number of edited images to generate.
        size (ImageSize | str): Size of the output images.
        response_format (ImageResponseFormat | str | None): Response format (e.g., 'url', 'b64_json'). Only supported
        by some models.
    """

    FILE_PREFIX: ClassVar[str] = "edited"

    group: Literal[NodeGroup.IMAGES] = NodeGroup.IMAGES
    name: str = "Image Edit"
    description: str = """Edit and modify existing images with text prompt and optional masking.

Key Capabilities:
- Image editing with natural language prompts
- Selective area editing using optional mask images
- Multiple variations generation (set n parameter)
- Configurable output sizes (256x256 to 1792x1024)
- URL or base64 JSON response formats

Usage Strategy:
- Provide clear, descriptive prompts for desired edits
- Use masks to target specific areas for modification
- Generate multiple variations to explore different results

Parameter Guide:
- prompt: Text description of desired edits (required)
- files: Image file/files to edit, auto-injected from agent's file store
- mask: Optional mask image to specify areas to edit
- n: Number of edited versions to generate
- size: Output dimensions (e.g., '1024x1024', '1792x1024')
- response_format: 'url' or 'b64_json' output format

Examples:
- {"prompt": "Add a sunset background behind the subject", "files": <source_image>}
- {"prompt": "Change the shirt color to blue", "files": <source_image>, "mask": <mask_image>}
- {"prompt": "Make the image more vibrant and colorful", "n": 3, "files": <source_image>}"""
    model: str = "gpt-image-1"
    connection: OpenAIConnection | None = None
    n: int | None = None
    size: ImageSize | str = ImageSize.SIZE_1024x1024
    response_format: ImageResponseFormat | str | None = Field(
        default=None,
        description="Response format (e.g., 'url', 'b64_json'). Only supported by some models. Will be dropped "
        "if not supported.",
    )
    error_handling: ErrorHandling = Field(default_factory=lambda: ErrorHandling(timeout_seconds=600))
    is_files_allowed: bool = True
    input_schema: ClassVar[type[ImageEditInputSchema]] = ImageEditInputSchema

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    _image_edit: Callable = PrivateAttr()

    def __init__(self, **kwargs):
        """Initialize the ImageEdit node.

        If neither client nor connection is provided, a new OpenAI connection is created.

        Args:
            **kwargs: Keyword arguments to initialize the node.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = OpenAIConnection()
        super().__init__(**kwargs)

        from litellm import image_edit

        self._image_edit = image_edit

    @property
    def edit_params(self) -> dict:
        """Get parameters for the image edit API call."""
        params = self.connection.conn_params.copy() if self.connection else {}
        if self.client:
            params["client"] = self.client
        if self.response_format is not None:
            response_format_value = (
                self.response_format.value
                if isinstance(self.response_format, ImageResponseFormat)
                else self.response_format
            )
            params["response_format"] = response_format_value

        if model_extra := getattr(self, "model_extra", None):
            extra = copy.deepcopy(model_extra)
            params.update(extra)

        return params

    def _prepare_image(self, image: list[io.BytesIO | bytes] | io.BytesIO | bytes) -> list[io.BytesIO] | io.BytesIO:
        """Prepare the image(s) for the API call.

        Args:
            image: Image as list of BytesIO/bytes (from FileStore), single BytesIO, or bytes.

        Returns:
            List of BytesIO file-like objects for API submission, or single BytesIO object.
        """
        if image is None:
            raise ValueError("No image provided. Please upload an image file.")

        if isinstance(image, list):
            if not image:
                raise ValueError("No image files found in storage.")
            return [prepare_single_image(img) for img in image]
        else:
            return prepare_single_image(image)

    def execute(self, input_data: ImageEditInputSchema, config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        """Execute the image editing.

        Args:
            input_data (ImageEditInputSchema): Input containing the image and prompt.
            config (RunnableConfig, optional): Configuration for the execution.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: Based on response_format:
                - URL: {"content": list[str], "files": list[BytesIO]} - list of image URLs and BytesIO file objects
                - B64_JSON: {"content": list[str], "files": list[BytesIO]} - list of created files data
                and BytesIO file objects
                Also includes "model" and "created" fields.
        """
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        size = self.size.value if isinstance(self.size, ImageSize) else self.size

        original_filenames = []
        raw_image = input_data.files
        images = raw_image if isinstance(raw_image, list) else [raw_image]
        for img in images:
            if img_name := getattr(img, "name", None):
                original_filenames.append(img_name)

        image = self._prepare_image(input_data.files)

        edit_kwargs = {
            "model": self.model,
            "image": image,
            "prompt": input_data.prompt,
            "size": size,
            "drop_params": True,
            **self.edit_params,
        }

        n = input_data.n or self.n
        if n:
            edit_kwargs["n"] = n

        if input_data.mask:
            mask = self._prepare_image(input_data.mask)
            edit_kwargs["mask"] = mask

        try:
            response = self._image_edit(**edit_kwargs)
        except Exception as e:
            logger.error(f"Node {self.name} - {self.id}: unexpected error occurred. Error: {str(e)}")
            raise ToolExecutionException(
                f"Node '{self.name}' encountered an unexpected error during image editing. "
                f"Error: {str(e)}. Please analyze the error and take appropriate action.",
                recoverable=True,
            )

        content = []
        files = []

        try:
            file_idx = 0
            for idx, img_data in enumerate(response.data):
                original_name = original_filenames[file_idx % len(original_filenames)] if original_filenames else None

                if img_url := getattr(img_data, ImageResponseFormat.URL.value, None):
                    content.append(img_url)
                    image_bytes = download_image_from_url(img_url)
                    file = create_image_file(
                        image_bytes, file_idx, original_name=original_name, prefix=self.FILE_PREFIX
                    )
                    files.append(file)
                    file_idx += 1

                elif img_b64 := getattr(img_data, ImageResponseFormat.B64_JSON.value, None):
                    image_bytes = base64.b64decode(img_b64)
                    file = create_image_file(
                        image_bytes, file_idx, original_name=original_name, prefix=self.FILE_PREFIX
                    )
                    content.append(f"{file.name} created")
                    files.append(file)
                    file_idx += 1
        except Exception as e:
            logger.error(f"Node {self.name} - {self.id}: failed to process response. Error: {str(e)}")
            raise ToolExecutionException(
                f"Node '{self.name}' failed to process edited image. " f"Error: {str(e)}. Please retry the request.",
                recoverable=True,
            )

        logger.debug(f"{self.name} edited image, generated {len(content)} result(s)")

        if self.is_optimized_for_agents:
            formatted_content = "## Edited Images\n\n"
            formatted_content += f"Created: {getattr(response, 'created', 'N/A')}\n"
            formatted_content += f"Count: {len(content)}\n\n"

            has_urls = content and isinstance(content[0], str) and content[0].startswith("http")
            if has_urls:
                for idx, url in enumerate(content):
                    formatted_content += f"### Edited Image {idx + 1}\n- URL: {url}\n\n"
            else:
                for idx, file_name in enumerate(content):
                    formatted_content += f"### Edited Image {idx + 1}\n- File: {file_name}\n\n"
            formatted_content += f"## Files Generated\n{len(files)} edited image file(s) available.\n"

            result = {"content": formatted_content}
        else:
            result = {
                "content": content,
                "created": getattr(response, "created", None),
            }

        result["files"] = files

        return result
