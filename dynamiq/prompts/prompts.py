import base64
import enum
import io
from abc import ABC, abstractmethod
from typing import Any

import filetype
from jinja2 import Environment, meta
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from dynamiq.utils import generate_uuid


class MessageRole(str, enum.Enum):
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"


class VisionMessageType(str, enum.Enum):
    TEXT = "text"
    IMAGE_URL = "image_url"


class VisionDetail(str, enum.Enum):
    AUTO = "auto"
    HIGH = "high"
    LOW = "low"


class Message(BaseModel):
    """
    Represents a message in a conversation.
    Attributes:
        content (str): The content of the message.
        role (MessageRole): The role of the message sender.
        metadata (dict | None): Additional metadata for the message, default is None.
    """

    content: str
    role: MessageRole = MessageRole.USER
    metadata: dict | None = None


class VisionMessageTextContent(BaseModel):
    """
    Represents a text message in a vision conversation.

    Attributes:
        type (VisionMessageType): The type of the message, default is "text".
        text (str): The text content of the message.
    """

    type: VisionMessageType = VisionMessageType.TEXT
    text: str


class VisionMessageImageURL(BaseModel):
    """
    Represents an image URL in a vision conversation.

    Attributes:
        url (str): The URL of the image.
        detail (VisionDetail): The detail level of the image, default is "auto".
    """

    url: str
    detail: VisionDetail = VisionDetail.AUTO


class VisionMessageImageContent(BaseModel):
    """
    Represents an image message in a vision conversation.

    Attributes:
        type (VisionMessageType): The type of the message, default is "image_url".
        image_url (VisionMessageImageURL): The image URL class.
    """
    type: VisionMessageType = VisionMessageType.IMAGE_URL
    image_url: VisionMessageImageURL


class VisionMessage(BaseModel):
    """
    Represents a vision message in a conversation.

    Attributes:
        content (list[VisionTextMessage | VisionImageMessage]): The content of the message.
        role (MessageRole): The role of the message sender.
    """

    content: list[VisionMessageTextContent | VisionMessageImageContent]
    role: MessageRole = MessageRole.USER

    def to_dict(self, **kwargs) -> dict:
        """
        Converts the message to a dictionary.

        Returns:
            dict: The message as a dictionary.
        """
        return self.model_dump(**kwargs)


class ToolFunctionParameters(BaseModel):
    type: str
    properties: dict[str, dict]
    required: list[str]


class ToolFunction(BaseModel):
    name: str
    description: str
    parameters: ToolFunctionParameters


class Tool(BaseModel):
    type: str = "function"
    function: ToolFunction


class BasePrompt(ABC, BaseModel):
    """
    Abstract base class for prompts.

    Attributes:
        id (str): Unique identifier for the prompt, generated using generate_uuid by default.
        version (str | None): Version of the prompt, optional.
    """

    id: str = Field(default_factory=generate_uuid)
    version: str | None = None

    @abstractmethod
    def format_messages(self, **kwargs) -> list[dict]:
        """
        Abstract method to format messages.

        Args:
            **kwargs: Arbitrary keyword arguments.

        Returns:
            list[dict]: A list of formatted messages as dictionaries.
        """
        pass

    @abstractmethod
    def format_tools(self, **kwargs) -> list[dict] | None:
        """
        Abstract method to format tools.

        Args:
            **kwargs: Arbitrary keyword arguments.

        Returns:
            list[dict]: A list of formatted tools as dictionaries.
        """
        pass


class Prompt(BasePrompt):
    """
    Concrete implementation of BasePrompt for handling both text and vision messages.

    Attributes:
        messages (list[Message | VisionMessage]): List of Message or VisionMessage objects
        representing the prompt.
        tools (list[dict[str, Any]]): List of functions for which the model may generate JSON inputs.
    """

    messages: list[Message | VisionMessage]
    tools: list[Tool] | None = None
    _Template: Any = PrivateAttr()

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def __init__(self, **data):
        super().__init__(**data)
        # Import and initialize Jinja2 Template here
        from jinja2 import Template

        self._Template = Template

    def get_parameters_for_template(self, template: str, env: Environment | None = None) -> set[str]:
        """
        Extracts set of parameters for template.

        Args:
            template (str): Template to find parameters for.
            env: (Environment, optional): jinja Environment object.

        Returns:
            set: Set of required parameters.
        """
        if not env:
            env = Environment(autoescape=True)
        # Parse the template to get its Abstract Syntax Tree
        ast = env.parse(template)

        # Find and return set of undeclared variables in the template
        return meta.find_undeclared_variables(ast)

    def get_required_parameters(self) -> set[str]:
        """Extracts set of parameters required for messages.

        Returns:
            set[str]: Set of parameter names.
        """
        parameters = set()

        env = Environment(autoescape=True)

        for msg in self.messages:
            if isinstance(msg, Message):
                parameters |= self.get_parameters_for_template(msg.content, env=env)
            elif isinstance(msg, VisionMessage):
                for content in msg.content:
                    if isinstance(content, VisionMessageTextContent):
                        parameters |= self.get_parameters_for_template(content.text, env=env)
                    elif isinstance(content, VisionMessageImageContent):
                        parameters |= self.get_parameters_for_template(content.image_url.url, env=env)
                    else:
                        raise ValueError(f"Invalid content type: {content.type}")
            else:
                raise ValueError(f"Invalid message type: {type(msg)}")

        return parameters

    def parse_image_url_parameters(self, url_template: str, kwargs: dict) -> None:
        """
        Converts image URL parameters in kwargs to Base64-encoded Data URLs if they contain image data.

        Args:
            url_template (str): Jinja template for the image URL.
            kwargs (dict): Dictionary of parameters to be used with the template.

        Raises:
            KeyError: If a required parameter is missing in kwargs.
            ValueError: If the file type cannot be determined or unsupported data type is provided.
        """
        template_params = self.get_parameters_for_template(url_template)

        for param in template_params:
            if param not in kwargs:
                raise KeyError(f"Missing required parameter: '{param}'")

            value = kwargs[param]

            # Initialize as unchanged; will be modified if image data is detected
            processed_value = value

            if isinstance(value, io.BytesIO):
                image_bytes = value.getvalue()
                extension = filetype.guess_extension(image_bytes)
                if not extension:
                    raise ValueError(f"Cannot determine file type for parameter '{param}'.")
                encoded_str = base64.b64encode(image_bytes).decode("utf-8")
                processed_value = f"data:image/{extension};base64,{encoded_str}"

            elif isinstance(value, bytes):
                extension = filetype.guess_extension(value)
                if not extension:
                    raise ValueError(f"Cannot determine file type for parameter '{param}'.")
                encoded_str = base64.b64encode(value).decode("utf-8")
                processed_value = f"data:image/{extension};base64,{encoded_str}"

            elif isinstance(value, str):
                pass  # No action needed; assuming it's a regular URL or already a Data URL

            else:
                # Unsupported data type for image parameter
                raise ValueError(f"Unsupported data type for parameter '{param}': {type(value)}")

            # Update the parameter with the processed value
            kwargs[param] = processed_value

    def format_messages(self, **kwargs) -> list[dict]:
        """
        Formats the messages in the prompt, rendering any templates.

        Args:
            **kwargs: Arbitrary keyword arguments used for template rendering.

        Returns:
            list[dict]: A list of formatted messages as dictionaries.
        """
        out: list[dict] = []
        for msg in self.messages:
            if isinstance(msg, Message):
                out.append(
                    Message(
                        role=msg.role,
                        content=self._Template(msg.content).render(**kwargs),
                    ).model_dump(exclude={"metadata"})
                )
            elif isinstance(msg, VisionMessage):
                out_msg_content = []
                for content in msg.content:
                    if isinstance(content, VisionMessageTextContent):
                        out_msg_content.append(
                            VisionMessageTextContent(
                                text=self._Template(content.text).render(**kwargs),
                            ).model_dump()
                        )
                    elif isinstance(content, VisionMessageImageContent):
                        self.parse_image_url_parameters(content.image_url.url, kwargs)
                        out_msg_content.append(
                            VisionMessageImageContent(
                                image_url=VisionMessageImageURL(
                                    url=self._Template(content.image_url.url).render(**kwargs),
                                    detail=content.image_url.detail,
                                )
                            ).model_dump()
                        )
                    else:
                        raise ValueError(f"Invalid content type: {content.type}")

                if len(out_msg_content) == 1 and out_msg_content[0]["type"] == VisionMessageType.TEXT:
                    out_msg_content = out_msg_content[0]["text"]

                out_msg = {
                    "content": out_msg_content,
                    "role": msg.role,
                }
                out.append(out_msg)
            else:
                raise ValueError(f"Invalid message type: {type(msg)}")

        return out

    def format_tools(self, **kwargs) -> list[dict] | None:
        out = None
        if self.tools:
            out = [tool.model_dump() for tool in self.tools]
        return out
