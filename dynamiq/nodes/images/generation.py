import base64
import io
from enum import Enum
from typing import Any, Callable, ClassVar, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from dynamiq.connections import AWS as AWSConnection
from dynamiq.connections import AzureAI as AzureAIConnection
from dynamiq.connections import Gemini as GeminiConnection
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.connections import VertexAI as VertexAIConnection
from dynamiq.nodes import ErrorHandling
from dynamiq.nodes.node import ConnectionNode, NodeGroup, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger


class ImageSize(str, Enum):
    """Supported image sizes for generation."""

    SIZE_256x256 = "256x256"
    SIZE_512x512 = "512x512"
    SIZE_1024x1024 = "1024x1024"
    SIZE_1024x1792 = "1024x1792"
    SIZE_1792x1024 = "1792x1024"


class ImageResponseFormat(str, Enum):
    """Response format for generated images."""

    URL = "url"
    B64_JSON = "b64_json"


def create_image_file(image_bytes: bytes, index: int = 0, original_name: str | None = None) -> io.BytesIO:
    """Create a properly configured BytesIO file from image bytes.

    Args:
        image_bytes: Raw image bytes.
        index: Index for naming multiple images.
        original_name: Optional original filename to base the new name on.
                       If provided, will create names like "original_0.png", "original_1.png".
                       If None, will use generic names like "image_0.png".

    Returns:
        BytesIO object with name and content_type attributes.
    """
    image_file = io.BytesIO(image_bytes)

    extension_to_mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
    }

    if original_name:
        from pathlib import Path

        base_name = Path(original_name).stem
        ext = Path(original_name).suffix or ".png"
        image_file.name = f"{base_name}_{index}{ext}"
        image_file.content_type = extension_to_mime.get(ext.lower(), "image/png")
    else:
        image_file.name = f"image_{index}.png"
        image_file.content_type = "image/png"

    return image_file


def download_image_from_url(url: str) -> bytes:
    """Download image from URL and return bytes.

    Args:
        url: URL of the image to download.

    Returns:
        Image bytes.
    """
    response = httpx.get(url, timeout=60.0)
    response.raise_for_status()
    return response.content


class ImageGenerationInputSchema(BaseModel):
    """Input schema for image generation."""

    prompt: str = Field(..., description="Text prompt describing the image to generate.")


class ImageGeneration(ConnectionNode):
    """
    Node for generating images using various AI models via LiteLLM.

    Supports multiple providers including OpenAI (DALL-E), Azure, Bedrock (Stability AI),
    Vertex AI, and more through LiteLLM's unified interface.

    Attributes:
        name (str): The name of the node.
        model (str): The model to use for image generation (e.g., 'dall-e-3', 'gpt-image-1',
            'bedrock/stability.stable-diffusion-xl-v0').
        connection (OpenAIConnection | GeminiConnection | VertexAIConnection | AWSConnection | AzureAIConnection):
        The connection to the API.
        n (int): Number of images to generate.
        size (ImageSize | str): Size of the generated images.
        quality (str | None): Quality of the generated images (e.g., 'standard', 'hd'). Only supported by some models.
        response_format (ImageResponseFormat | str | None): Response format (e.g., 'url', 'b64_json').
        Only supported by some models.
    """

    group: Literal[NodeGroup.IMAGES] = NodeGroup.IMAGES
    name: str = "Image Generation"
    model: str = "dall-e-3"
    connection: OpenAIConnection | GeminiConnection | VertexAIConnection | AWSConnection | AzureAIConnection | None = (
        None
    )
    n: int | None = None
    size: ImageSize | str = ImageSize.SIZE_1024x1024
    quality: str | None = Field(
        default=None,
        description="Image quality (e.g., 'standard', 'hd'). Only supported by some models. Will be dropped "
        "if not supported.",
    )
    response_format: ImageResponseFormat | str | None = Field(
        default=None,
        description="Response format (e.g., 'url', 'b64_json'). Only supported by some models. Will be dropped "
        "if not supported.",
    )
    error_handling: ErrorHandling = Field(default_factory=lambda: ErrorHandling(timeout_seconds=600))
    input_schema: ClassVar[type[ImageGenerationInputSchema]] = ImageGenerationInputSchema

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    _image_generation: Callable = PrivateAttr()

    def __init__(self, **kwargs):
        """Initialize the ImageGeneration node.

        If neither client nor connection is provided, a new OpenAI connection is created.

        Args:
            **kwargs: Keyword arguments to initialize the node.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = OpenAIConnection()
        super().__init__(**kwargs)

        from litellm import image_generation

        self._image_generation = image_generation

    @property
    def generation_params(self) -> dict:
        """Get parameters for the image generation API call."""
        params = self.connection.conn_params.copy() if self.connection else {}
        if self.client:
            params["client"] = self.client
        if self.quality is not None:
            params["quality"] = self.quality
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

    def execute(
        self, input_data: ImageGenerationInputSchema, config: RunnableConfig = None, **kwargs
    ) -> dict[str, Any]:
        """Execute the image generation.

        Args:
            input_data (ImageGenerationInputSchema): Input containing the prompt.
            config (RunnableConfig, optional): Configuration for the execution.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: Based on actual response fields:
                - If response has "url": {"content": list[str], "files": list[BytesIO]} - list of image URLs and
                BytesIO file objects
                - If response has "b64_json": {"content": list[str], "files": list[BytesIO]} - list of created files
                data and BytesIO file objects
        """
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        size = self.size.value if isinstance(self.size, ImageSize) else self.size

        api_params = {
            "model": self.model,
            "prompt": input_data.prompt,
            "n": self.n,
            "size": size,
            "drop_params": True,
            **self.generation_params,
        }

        response = self._image_generation(**api_params)

        content = []
        files = []

        for idx, img_data in enumerate(response.data):
            if hasattr(img_data, "url") and img_data.url:
                content.append(img_data.url)
                image_bytes = download_image_from_url(img_data.url)
                file = create_image_file(image_bytes, idx)
                files.append(file)

            elif hasattr(img_data, "b64_json") and img_data.b64_json:
                image_bytes = base64.b64decode(img_data.b64_json)
                file = create_image_file(image_bytes, idx)
                content.append(f"{file.name} created")
                files.append(file)

        logger.debug(f"{self.name} generated {len(content)} image(s)")

        if self.is_optimized_for_agents:
            formatted_content = "## Generated Images\n\n"
            formatted_content += f"Created: {getattr(response, 'created', 'N/A')}\n"
            formatted_content += f"Count: {len(content)}\n\n"

            has_urls = content and isinstance(content[0], str) and content[0].startswith("http")
            if has_urls:
                for idx, url in enumerate(content):
                    formatted_content += f"### Image {idx + 1}\n- URL: {url}\n\n"
            else:
                for idx, file_name in enumerate(content):
                    formatted_content += f"### Image {idx + 1}\n- File: {file_name}\n\n"
            formatted_content += f"## Files Generated\n{len(files)} image file(s) available.\n"

            result = {"content": formatted_content}
        else:
            result = {
                "content": content,
                "created": getattr(response, "created", None),
            }

        result["files"] = files

        return result
