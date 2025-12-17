import base64
import io
from typing import Any, Callable, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes import ErrorHandling
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode, NodeGroup, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

from .edit import prepare_single_image
from .generation import ImageResponseFormat, ImageSize, create_image_file, download_image_from_url


class ImageVariationInputSchema(BaseModel):
    """Input schema for image variation."""

    image: list[io.BytesIO | bytes] | io.BytesIO | bytes = Field(
        ...,
        description="The image to create variations of. Auto-injected from agent's file store.",
        json_schema_extra={"map_from_storage": True, "is_accessible_to_agent": False},
    )
    n: int | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ImageVariation(ConnectionNode):
    """
    Node for creating variations of images using AI models.

    Takes an existing image and generates variations of it.

    Attributes:
        name (str): The name of the node.
        model (str): The model to use for image variation (e.g., 'dall-e-2').
        connection (OpenAIConnection): The connection to the API.
        n (int): Number of variations to generate.
        size (ImageSize | str): Size of the output images.
        response_format (ImageResponseFormat | str | None): Response format (e.g., 'url', 'b64_json'). Only supported
        by some models.
    """

    group: Literal[NodeGroup.IMAGES] = NodeGroup.IMAGES
    name: str = "Image Variation"
    description: str = """Create AI-generated variations of existing images while preserving core composition.

Key Capabilities:
- Generate creative variations from source images
- Maintain overall composition and subject matter
- Multiple variation generation (set n parameter)
- Configurable output sizes (256x256 to 1024x1024)
- URL or base64 JSON response formats
- Auto-injection from agent's file store
- Preserves original filenames in outputs

Usage Strategy:
- Provide a clear, high-quality source image
- Generate multiple variations to explore creative alternatives
- Use for style exploration, A/B testing, or creative ideation
- Integrate with agents for dynamic image transformation workflows

Parameter Guide:
- image: Source image file, auto-injected from agent's file store (required)
- n: Number of variations to generate (default: 1)
- size: Output dimensions (e.g., '1024x1024', '512x512')
- response_format: 'url' or 'b64_json' output format

Examples:
- {"image": <source_image>} - Single variation of uploaded image
- {"image": <source_image>, "n": 5} - Generate 5 different variations
- {"image": <source_image>, "size": "512x512"} - Smaller output size"""
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
    input_schema: ClassVar[type[ImageVariationInputSchema]] = ImageVariationInputSchema

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    _image_variation: Callable = PrivateAttr()

    def __init__(self, **kwargs):
        """Initialize the Image Variation node.

        If neither client nor connection is provided, a new OpenAI connection is created.

        Args:
            **kwargs: Keyword arguments to initialize the node.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = OpenAIConnection()
        super().__init__(**kwargs)

        from litellm import image_variation

        self._image_variation = image_variation

    @property
    def variation_params(self) -> dict:
        """Get parameters for the image variation API call."""
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

    def execute(self, input_data: ImageVariationInputSchema, config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        """Execute the image variation generation.

        Args:
            input_data (ImageVariationInputSchema): Input containing the image.
            config (RunnableConfig, optional): Configuration for the execution.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: Based on response_format:
                - URL: {"content": list[str], "files": list[BytesIO]} - list of image URLs and BytesIO file objects
                - B64_JSON: {"content": list[str], "files": list[BytesIO]} - list of created files data
                and BytesIO file objects
        """
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        size = self.size.value if isinstance(self.size, ImageSize) else self.size
        raw_image = input_data.image
        if isinstance(raw_image, list):
            if not raw_image:
                raise ValueError("No image provided. List is empty.")
            if len(raw_image) > 1:
                logger.warning("Multiple images provided, using first image for variation.")
            raw_image = raw_image[0]
        original_filename = None
        if hasattr(raw_image, "name") and raw_image.name:
            original_filename = raw_image.name
        elif isinstance(raw_image, io.BytesIO) and hasattr(raw_image, "name") and raw_image.name:
            original_filename = raw_image.name

        image = prepare_single_image(raw_image)

        api_params = {
            "model": self.model,
            "image": image,
            "size": size,
            "drop_params": True,
            **self.variation_params,
        }

        n = input_data.n or self.n
        if n:
            api_params["n"] = n
        try:
            response = self._image_variation(**api_params)
        except Exception as e:
            logger.error(f"Node {self.name} - {self.id}: unexpected error occurred. Error: {str(e)}")
            raise ToolExecutionException(
                f"Node '{self.name}' encountered an unexpected error during image variation generation. "
                f"Error: {str(e)}. Please analyze the error and take appropriate action.",
                recoverable=True,
            )

        content = []
        files = []

        for idx, img_data in enumerate(response.data):
            if img_url := getattr(img_data, ImageResponseFormat.URL.value, None):
                content.append(img_url)
                image_bytes = download_image_from_url(img_url)
                file = create_image_file(image_bytes, idx, original_name=original_filename)
                files.append(file)

            elif img_b64 := getattr(img_data, ImageResponseFormat.B64_JSON.value, None):
                image_bytes = base64.b64decode(img_b64)
                file = create_image_file(image_bytes, idx, original_name=original_filename)
                content.append(f"{file.name} created")
                files.append(file)

        logger.debug(f"{self.name} generated {len(content)} variation(s)")

        if self.is_optimized_for_agents:
            formatted_content = "## Image Variations\n\n"
            formatted_content += f"Created: {getattr(response, 'created', 'N/A')}\n"
            formatted_content += f"Count: {len(content)}\n\n"

            has_urls = content and isinstance(content[0], str) and content[0].startswith("http")
            if has_urls:
                for idx, url in enumerate(content):
                    formatted_content += f"### Variation {idx + 1}\n- URL: {url}\n\n"
            else:
                for idx, file_name in enumerate(content):
                    formatted_content += f"### Variation {idx + 1}\n- File: {file_name}\n\n"
            formatted_content += f"## Files Generated\n{len(files)} variation file(s) available.\n"

            result = {"content": formatted_content}
        else:
            result = {
                "content": content,
                "created": getattr(response, "created", None),
            }

        result["files"] = files

        return result
