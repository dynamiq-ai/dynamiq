import base64
import io
from typing import Any, Callable, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes import ErrorHandling
from dynamiq.nodes.node import ConnectionNode, NodeGroup, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

from .generation import ImageResponseFormat, ImageSize, create_image_file, download_image_from_url


class ImageEditInputSchema(BaseModel):
    """Input schema for image editing."""

    prompt: str = Field(..., description="Text prompt describing the desired edits.")
    image: list[io.BytesIO | bytes] | list[bytes] | io.BytesIO | bytes | None = Field(
        default=None,
        description="The image(s) to edit. Can be a single image or a list of images. Auto-injected from agent's "
        "file store.",
        json_schema_extra={"map_from_storage": True, "is_accessible_to_agent": False},
    )
    mask: io.BytesIO | bytes | None = Field(
        default=None,
        description="Optional mask image indicating areas to edit.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ImageEdit(ConnectionNode):
    """
    Node for editing images using AI models via LiteLLM.

    Takes an existing image and a text prompt to generate edited versions.
    Optionally accepts a mask to specify which areas to modify.

    Attributes:
        name (str): The name of the node.
        model (str): The model to use for image editing (e.g., 'dall-e-2', 'gpt-image-1').
        connection (BaseConnection): The connection to the API.
        n (int): Number of edited images to generate.
        size (ImageSize | str): Size of the output images.
        response_format (ImageResponseFormat | str | None): Response format (e.g., 'url', 'b64_json'). Only supported
        by some models.
    """

    group: Literal[NodeGroup.IMAGES] = NodeGroup.IMAGES
    name: str = "Image Edit"
    model: str = "dall-e-2"
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

        if hasattr(self, "__pydantic_extra__") and self.__pydantic_extra__:
            import copy

            extra = copy.deepcopy(self.__pydantic_extra__)
            params.update(extra)

        return params

    def _prepare_image(
        self, image: list[io.BytesIO] | list[bytes] | io.BytesIO | bytes | None
    ) -> list[io.BytesIO] | io.BytesIO:
        """Prepare the image(s) for the API call.

        Args:
            image: Image as list of BytesIO/bytes (from FileStore), single BytesIO, or bytes.

        Returns:
            List of BytesIO file-like objects for API submission, or single BytesIO object.
        """
        if image is None:
            raise ValueError("No image provided. Please upload an image file.")

        def prepare_single_image(img) -> io.BytesIO:
            if isinstance(img, bytes):
                image_bytes = img
            elif isinstance(img, io.BytesIO):
                img.seek(0)
                image_bytes = img.read()
            elif hasattr(img, "read"):
                img.seek(0)
                image_bytes = img.read()
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")

            from PIL import Image

            img_obj = Image.open(io.BytesIO(image_bytes))
            original_format = img_obj.format or "PNG"
            if img_obj.mode not in ("RGBA", "LA", "L"):
                img_obj = img_obj.convert("RGBA")

            output_bytes = io.BytesIO()
            img_obj.save(output_bytes, format=original_format)
            output_bytes.seek(0)
            return output_bytes

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
        raw_image = input_data.image
        if isinstance(raw_image, list):
            for img in raw_image:
                if hasattr(img, "name") and img.name:
                    original_filenames.append(img.name)
                elif isinstance(img, io.BytesIO) and hasattr(img, "name") and img.name:
                    original_filenames.append(img.name)
        else:
            if hasattr(raw_image, "name") and raw_image.name:
                original_filenames.append(raw_image.name)
            elif isinstance(raw_image, io.BytesIO) and hasattr(raw_image, "name") and raw_image.name:
                original_filenames.append(raw_image.name)

        image = self._prepare_image(input_data.image)

        edit_kwargs = {
            "model": self.model,
            "image": image,
            "prompt": input_data.prompt,
            "n": self.n,
            "size": size,
            "drop_params": True,
            **self.edit_params,
        }

        mask = self._prepare_image(input_data.mask) if input_data.mask else None
        if mask:
            edit_kwargs["mask"] = mask
        response = self._image_edit(**edit_kwargs)

        content = []
        files = []

        file_idx = 0
        for idx, img_data in enumerate(response.data):
            original_name = original_filenames[file_idx % len(original_filenames)] if original_filenames else None

            if hasattr(img_data, "url") and img_data.url:
                content.append(img_data.url)
                image_bytes = download_image_from_url(img_data.url)
                file = create_image_file(image_bytes, file_idx, original_name=original_name)
                files.append(file)
                file_idx += 1

            elif hasattr(img_data, "b64_json") and img_data.b64_json:
                image_bytes = base64.b64decode(img_data.b64_json)
                file = create_image_file(image_bytes, file_idx, original_name=original_name)
                content.append(f"{file.name} created")
                files.append(file)
                file_idx += 1

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
