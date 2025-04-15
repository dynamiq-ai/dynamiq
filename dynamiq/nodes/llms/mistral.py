from dynamiq.connections import Mistral as MistralConnection
from dynamiq.nodes.llms.base import BaseLLM
from dynamiq.prompts import MessageRole


class Mistral(BaseLLM):
    """Mistral LLM node.

    This class provides an implementation for the Mistral Language Model node.

    Attributes:
        connection (MistralConnection | None): The connection to use for the Mistral LLM.
        MODEL_PREFIX (str): The prefix for the Mistral model name.
    """
    connection: MistralConnection | None = None
    MODEL_PREFIX = "mistral/"

    def __init__(self, **kwargs):
        """Initialize the Mistral LLM node.

        Args:
            **kwargs: Additional keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = MistralConnection()
        super().__init__(**kwargs)

    def get_messages(
        self,
        prompt,
        input_data,
    ) -> list[dict]:
        """
        Format and filter message parameters based on provider requirements.
        Override this in provider-specific subclasses.
        """
        messages = prompt.format_messages(**dict(input_data))
        formatted_messages = []
        for i, msg in enumerate(messages):
            msg_copy = msg.copy()

            is_last_message = i == len(messages) - 1
            if is_last_message and msg_copy["role"] == MessageRole.ASSISTANT.value:
                msg_copy["prefix"] = True

            formatted_messages.append(msg_copy)

        return formatted_messages
