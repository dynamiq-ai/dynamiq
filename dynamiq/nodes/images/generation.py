import base64
import copy
import io
from enum import Enum
from pathlib import Path
from typing import Any, Callable, ClassVar, Literal

import filetype
import requests
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from dynamiq.connections import AWS as AWSConnection
from dynamiq.connections import AzureAI as AzureAIConnection
from dynamiq.connections import Gemini as GeminiConnection
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.connections import VertexAI as VertexAIConnection
from dynamiq.nodes import ErrorHandling
from dynamiq.nodes.agents.exceptions import ToolExecutionException
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


def create_image_file(
    image_bytes: bytes, index: int = 0, original_name: str | None = None, prefix: str | None = None
) -> io.BytesIO:
    """Create a properly configured BytesIO file from image bytes.

    Args:
        image_bytes: Raw image bytes.
        index: Index for naming multiple images.
        original_name: Optional original filename to base the new name on.
                       If provided, will create names like "original_0.png", "original_1.png".
                       If None, will use generic names like "generated_image_0.png" or "image_0.png" if prefix is None.
        prefix: Optional prefix for the filename (e.g., "generated", "edited", "variation").

    Returns:
        BytesIO object with name and content_type attributes.
    """
    image_file = io.BytesIO(image_bytes)

    kind = filetype.guess(image_bytes)
    ext = f".{kind.extension}" if kind else ".png"
    prefix_str = f"{prefix}_" if prefix else ""
    if original_name:
        base_name = Path(original_name).stem
        image_file.name = f"{prefix_str}{base_name}_{index}{ext}"
    else:
        image_file.name = f"{prefix_str}image_{index}{ext}"

    image_file.content_type = kind.mime if kind else "image/png"
    return image_file


def download_image_from_url(url: str) -> bytes:
    """Download image from URL and return bytes.

    Args:
        url: URL of the image to download.

    Returns:
        Image bytes.
    """
    response = requests.get(url, timeout=60.0)
    response.raise_for_status()
    return response.content


class ImageGenerationInputSchema(BaseModel):
    """Input schema for image generation."""

    prompt: str = Field(..., description="Text prompt describing the image to generate.")
    n: int | None = None


class ImageGeneration(ConnectionNode):
    """
    Node for generating images using various AI models.

    Supports multiple providers including OpenAI (DALL-E), Azure, Bedrock (Stability AI),
    Vertex AI, and more through a unified interface.

    Attributes:
        FILE_PREFIX (str): Prefix for new file names. Default to "generated".
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

    FILE_PREFIX: ClassVar[str] = "generated"

    group: Literal[NodeGroup.IMAGES] = NodeGroup.IMAGES
    name: str = "Image Generation"
    description: str = """Generate images from text prompt using image generation models.

Key Capabilities:
- Text-to-image generation with natural language prompts
- Multi-provider support (OpenAI, Azure, AWS Bedrock, Vertex AI, Gemini)
- Multiple image generation (set n parameter)
- Configurable sizes (256x256 to 1792x1024)
- Quality control (standard/hd for supported models)
- URL or base64 JSON response formats

Usage Strategy:
- Write detailed, descriptive prompts for best results
- Specify artistic style, composition, and key elements clearly
- Generate multiple variations to explore creative options
- Use quality='hd' for high-detail requirements

Parameter Guide:
- prompt: Detailed text description of the image to generate (required)
- n: Number of images to create (default: 1)
- size: Output dimensions (e.g., '1024x1024', '1024x1792', '1792x1024')
- quality: 'standard' or 'hd' (model-dependent)
- response_format: 'url' or 'b64_json' output format

Examples:
- {"prompt": "A serene mountain landscape at sunset, photorealistic"}
- {"prompt": "Modern minimalist logo for tech startup, blue and white", "n": 3}
- {"prompt": "Abstract art with vibrant colors", "size": "1792x1024", "quality": "hd"}"""
    model: str = "gpt-image-1"
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

        if model_extra := getattr(self, "model_extra", None):
            extra = copy.deepcopy(model_extra)
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
            "size": size,
            "drop_params": True,
            **self.generation_params,
        }

        n = input_data.n or self.n
        if n:
            api_params["n"] = n

        try:
            response = self._image_generation(**api_params)
        except Exception as e:
            logger.error(f"Node {self.name} - {self.id}: unexpected error occurred. Error: {str(e)}")
            raise ToolExecutionException(
                f"Node '{self.name}' encountered an unexpected error during image generation. "
                f"Error: {str(e)}. Please analyze the error and take appropriate action.",
                recoverable=True,
            )

        content = []
        files = []

        try:
            for idx, img_data in enumerate(response.data):
                if img_url := getattr(img_data, ImageResponseFormat.URL.value, None):
                    content.append(img_url)
                    image_bytes = download_image_from_url(img_url)
                    file = create_image_file(image_bytes, idx, prefix=self.FILE_PREFIX)
                    files.append(file)

                elif img_b64 := getattr(img_data, ImageResponseFormat.B64_JSON.value, None):
                    image_bytes = base64.b64decode(img_b64)
                    file = create_image_file(image_bytes, idx, prefix=self.FILE_PREFIX)
                    content.append(f"{file.name} created")
                    files.append(file)
        except Exception as e:
            logger.error(f"Node {self.name} - {self.id}: failed to process response. Error: {str(e)}")
            raise ToolExecutionException(
                f"Node '{self.name}' failed to process generated image. " f"Error: {str(e)}. Please retry the request.",
                recoverable=True,
            )

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
