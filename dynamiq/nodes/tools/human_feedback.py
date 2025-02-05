from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.nodes import NodeGroup
from dynamiq.nodes.node import Node, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.types.feedback import FeedbackMethod
from dynamiq.types.streaming import StreamingEventMessage
from dynamiq.utils.logger import logger


class HFStreamingInputEventMessageData(BaseModel):
    content: str


class HFStreamingInputEventMessage(StreamingEventMessage):
    data: HFStreamingInputEventMessageData


class HFStreamingOutputEventMessageData(BaseModel):
    prompt: str


class HFStreamingOutputEventMessage(StreamingEventMessage):
    data: HFStreamingOutputEventMessageData


class InputMethodCallable(ABC):
    """
    Abstract base class for input methods.

    This class defines the interface for various input methods that can be used
    to gather user input in the HumanFeedbackTool.
    """

    @abstractmethod
    def get_input(self, prompt: str, **kwargs) -> str:
        """
        Get input from the user.

        Args:
            prompt (str): The prompt to display to the user.

        Returns:
            str: The user's input.
        """
        pass


class OutputMethodCallable(ABC):
    """
    Abstract base class for sending message.

    This class defines the interface for various output methods that can be used
    to send user information in the MessageSenderTool.
    """

    @abstractmethod
    def send_message(self, message: str, **kwargs) -> None:
        """
        Sends message to the user

        Args:
            message (str): The message to send to the user.
        """

        pass


class HumanFeedbackInputSchema(BaseModel):
    input: str = Field(..., description="Parameter to provide a question to the user.")


class HumanFeedbackTool(Node):
    """
    A tool for gathering user information through human feedback.

    This tool prompts the user for input and returns the response. It should be used to check actual
    information from the user or to gather additional input during a process.

    Attributes:
        group (Literal[NodeGroup.TOOLS]): The group the node belongs to.
        name (str): The name of the tool.
        description (str): A brief description of the tool's purpose.
        input_method (FeedbackMethod | InputMethodCallable): The method used to gather user input.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "Human Feedback Tool"
    description: str = (
        "Tool to gather feedback from the user. Use it to check actual information or get additional input."
    )
    input_method: FeedbackMethod | InputMethodCallable = FeedbackMethod.CONSOLE
    input_schema: ClassVar[type[HumanFeedbackInputSchema]] = HumanFeedbackInputSchema

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def input_method_console(self, prompt: str) -> str:
        """
        Get input from the user using the console input method.

        Args:
            prompt (str): The prompt to display to the user.

        Returns:
            str: The user's input.
        """
        return input(prompt)

    def input_method_streaming(self, prompt: str, config: RunnableConfig, **kwargs) -> str:
        """
        Get input from the user using the queue streaming input method.

        Args:
            prompt (str): The prompt to display to the user.
            config (RunnableConfig, optional): The configuration for the runnable. Defaults to None.

        Returns:
            str: The user's input.
        """
        logger.debug(f"Tool {self.name} - {self.id}: started with prompt {prompt}")
        event = HFStreamingOutputEventMessage(
            wf_run_id=config.run_id,
            entity_id=self.id,
            data=HFStreamingOutputEventMessageData(prompt=prompt),
            event=self.streaming.event,
        )
        logger.debug(f"Tool {self.name} - {self.id}: sending output event {event}")
        self.run_on_node_execute_stream(callbacks=config.callbacks, event=event, **kwargs)
        event = self.get_input_streaming_event(
            event_msg_type=HFStreamingInputEventMessage,
            event=self.streaming.event,
            config=config,
        )
        logger.debug(f"Tool {self.name} - {self.id}: received input event {event}")

        return event.data.content

    def execute(
        self, input_data: HumanFeedbackInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Execute the tool with the provided input data and configuration.

        This method prompts the user for input using the specified input method and returns the result.

        Args:
            input_data (dict[str, Any]): The input data containing the prompt for the user.
            config (RunnableConfig, optional): The configuration for the runnable. Defaults to None.
            **kwargs: Additional keyword arguments to be passed to the node execute run.

        Returns:
            dict[str, Any]: A dictionary containing the user's input under the 'content' key.

        Raises:
            ValueError: If the input_data does not contain an 'input' key.
        """
        logger.debug(f"Tool {self.name} - {self.id}: started with input data {input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        input_text = input_data.input
        if isinstance(self.input_method, FeedbackMethod):
            if self.input_method == FeedbackMethod.CONSOLE:
                result = self.input_method_console(input_text)
            elif self.input_method == FeedbackMethod.STREAM:
                streaming = getattr(config.nodes_override.get(self.id), "streaming", None) or self.streaming
                if not streaming.input_streaming_enabled:
                    raise ValueError(
                        f"'{FeedbackMethod.STREAM}' input method requires enabled input and output streaming."
                    )

                result = self.input_method_streaming(prompt=input_text, config=config, **kwargs)
            else:
                raise ValueError(f"Unsupported input method: {self.input_method}")
        else:
            result = self.input_method.get_input(input_text)

        logger.debug(f"Tool {self.name} - {self.id}: finished with result {result}")
        return {"content": result}


class MessageSenderInputSchema(BaseModel):
    input: str = Field(..., description="Parameter to provide a message to the user.")


class MessageSenderTool(Node):
    """
    A tool for sending messsages.

    Attributes:
        group (Literal[NodeGroup.TOOLS]): The group the node belongs to.
        name (str): The name of the tool.
        description (str): A brief description of the tool's purpose.
        output_method (FeedbackMethod | InputMethodCallable): The method used to gather user input.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "Message Sender Tool"
    description: str = "Tool can be used send messages to the user."
    output_method: FeedbackMethod | OutputMethodCallable = FeedbackMethod.CONSOLE
    input_schema: ClassVar[type[MessageSenderInputSchema]] = MessageSenderInputSchema

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def output_method_console(self, prompt: str) -> None:
        """
        Sends message to console.

        Args:
            prompt (str): The prompt to display to the user.
        """
        print(prompt)

    def output_method_streaming(self, prompt: str, config: RunnableConfig, **kwargs) -> None:
        """
        Sends message using streaming method.

        Args:
            prompt (str): The prompt to display to the user.
            config (RunnableConfig, optional): The configuration for the runnable. Defaults to None.
        """

        event = HFStreamingOutputEventMessage(
            wf_run_id=config.run_id,
            entity_id=self.id,
            data=HFStreamingOutputEventMessageData(prompt=prompt),
            event=self.streaming.event,
        )
        logger.debug(f"Tool {self.name} - {self.id}: sending output event {event}")
        self.run_on_node_execute_stream(callbacks=config.callbacks, event=event, **kwargs)

    def execute(
        self, input_data: MessageSenderInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Execute the tool with the provided input data and configuration.

        This method prompts the user for input using the specified input method and returns the result.

        Args:
            input_data (dict[str, Any]): The input data containing the prompt for the user.
            config (RunnableConfig, optional): The configuration for the runnable. Defaults to None.
            **kwargs: Additional keyword arguments to be passed to the node execute run.

        Returns:
            dict[str, Any]: A dictionary containing the user's input under the 'content' key.

        Raises:
            ValueError: If the input_data does not contain an 'input' key.
        """
        logger.debug(f"Tool {self.name} - {self.id}: started with input data {input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        input_text = input_data.input
        if isinstance(self.output_method, FeedbackMethod):
            if self.output_method == FeedbackMethod.CONSOLE:
                self.output_method_console(input_text)
            elif self.output_method == FeedbackMethod.STREAM:
                streaming = getattr(config.nodes_override.get(self.id), "streaming", None) or self.streaming
                if not streaming.input_streaming_enabled:
                    raise ValueError(
                        f"'{FeedbackMethod.STREAM}' input method requires enabled input and output streaming."
                    )

                self.output_method_streaming(prompt=input_text, config=config, **kwargs)
            else:
                raise ValueError(f"Unsupported feedback method: {self.output_method}")
        else:
            self.output_method.send_message(input_text)

        logger.debug(f"Tool {self.name} - {self.id}: finished")
        return {"content": f"Message {input_text} was sent."}
