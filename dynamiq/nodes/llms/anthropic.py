from dynamiq.connections import Anthropic as AnthropicConnection
from dynamiq.nodes.llms.base import BaseLLM
from dynamiq.prompts.prompts import VisionMessageType
from dynamiq.utils.logger import logger


class Anthropic(BaseLLM):
    """Anthropic LLM node.

    This class provides an implementation for the Anthropic Language Model node.

    Attributes:
        connection (AnthropicConnection | None): The connection to use for the Anthropic LLM.
    """
    connection: AnthropicConnection | None = None
    MODEL_PREFIX = "anthropic/"

    def __init__(self, **kwargs):
        """Initialize the Anthropic LLM node.

        Args:
            **kwargs: Additional keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = AnthropicConnection()
        super().__init__(**kwargs)

    @staticmethod
    def _convert_non_image_to_file_content(messages: list[dict]) -> list[dict]:
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue

            new_content = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == VisionMessageType.IMAGE_URL and "image_url" in item:
                    url = item["image_url"].get("url", "")
                    if url.startswith("data:") and not url.startswith("data:image/"):
                        logger.debug("Anthropic: converting non-image image_url to file content format")
                        new_content.append(
                            {
                                "type": VisionMessageType.FILE,
                                "file": {"file_data": url},
                            }
                        )
                    else:
                        new_content.append(item)
                else:
                    new_content.append(item)

            message["content"] = new_content

        return messages

    def get_messages(self, prompt, input_data) -> list[dict]:
        """
        Format messages and convert non-image files to Anthropic file content format.
        """
        messages = super().get_messages(prompt, input_data)
        return self._convert_non_image_to_file_content(messages)
