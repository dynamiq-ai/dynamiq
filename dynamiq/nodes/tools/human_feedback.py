from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal
from queue import Queue

from jinja2 import Template
from pydantic import BaseModel, ConfigDict, model_validator

from dynamiq.nodes import NodeGroup
from dynamiq.nodes.node import Node, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.types.feedback import FeedbackMethod
from dynamiq.types.streaming import StreamingEntitySource, StreamingEventMessage
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
    model_config = ConfigDict(extra="allow")


class HumanFeedbackTool(Node):
    """
    A tool for gathering user information through human feedback.

    This tool automatically adapts to the environment:
    - Console: Uses input() for local development
    - Streaming: Uses WebSocket/queue for UI applications
    - Auto-detection: Automatically chooses method based on configuration

    Attributes:
        group (Literal[NodeGroup.TOOLS]): The group the node belongs to.
        name (str): The name of the tool.
        description (str): A brief description of the tool's purpose.
        msg_template (str): Template of message to send.
        input_method (FeedbackMethod | InputMethodCallable): The method used to gather user input.
        auto_detect (bool): Automatically detect the best input method based on environment.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "Human Feedback Tool"
    description: str = """Collects human feedback during workflow execution for validation and decision-making.

Key Capabilities:
- Workflow pause for human input collection
- Console and streaming interface support
- Customizable message templates with parameter substitution
- Integration with workflow orchestrators and streaming systems

Usage Strategy:
- Use for content validation before publication
- Implement decision confirmation for high-stakes operations
- Create quality assurance checkpoints in automated workflows
- Collect user preferences and customization inputs

Parameter Guide:
- msg_template: Jinja2 template for message formatting
Examples:
- {"msg_template": "Please review this email draft before sending"}
"""
    input_method: FeedbackMethod | InputMethodCallable = FeedbackMethod.CONSOLE
    input_schema: ClassVar[type[HumanFeedbackInputSchema]] = HumanFeedbackInputSchema
    msg_template: str = "{{input}}"
    auto_detect: bool = True
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def update_description(self):
        msg_template = self.msg_template
        self.description += (
            f"\nThis is the template of message to send: '{msg_template}'."
            " Parameters will be substituted based on the provided input data."
        )
        return self

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

        streaming = getattr(config.nodes_override.get(self.id), "streaming", None) or self.streaming

        event = HFStreamingOutputEventMessage(
            wf_run_id=config.run_id,
            entity_id=self.id,
            data=HFStreamingOutputEventMessageData(prompt=prompt),
            event=streaming.event,
            source=StreamingEntitySource(
                name=self.name,
                group=self.group,
                type=self.type,
            ),
        )
        logger.debug(f"Tool {self.name} - {self.id}: sending output event {event}")
        self.run_on_node_execute_stream(callbacks=config.callbacks, event=event, **kwargs)
        event = self.get_input_streaming_event(
            event_msg_type=HFStreamingInputEventMessage,
            event=streaming.event,
            config=config,
        )
        logger.debug(f"Tool {self.name} - {self.id}: received input event {event}")

        return event.data.content

    def _detect_input_method(self, config: RunnableConfig) -> FeedbackMethod:
        """
        Automatically detect the best input method based on the environment.
        
        Args:
            config (RunnableConfig): The configuration for the runnable.
            
        Returns:
            FeedbackMethod: The detected input method.
        """
        streaming = getattr(config.nodes_override.get(self.id), "streaming", None) or self.streaming
        
        if streaming and streaming.enabled and hasattr(streaming, 'input_queue') and streaming.input_queue:
            return FeedbackMethod.STREAM
        
        return FeedbackMethod.CONSOLE
    
    def _ensure_streaming_config(self, config: RunnableConfig) -> None:
        """
        Ensure streaming configuration is properly set up when using STREAM method.
        
        Args:
            config (RunnableConfig): The configuration for the runnable.
        """
        streaming = getattr(config.nodes_override.get(self.id), "streaming", None) or self.streaming
        
        if not streaming:
            from dynamiq.types.streaming import StreamingConfig
            streaming = StreamingConfig(enabled=True, input_queue=Queue())
            
            if not config.nodes_override:
                config.nodes_override = {}
            if self.id not in config.nodes_override:
                from dynamiq.runnables.base import NodeRunnableConfig
                config.nodes_override[self.id] = NodeRunnableConfig()
            config.nodes_override[self.id].streaming = streaming
        elif not streaming.enabled:
            streaming.enabled = True
        
        if not hasattr(streaming, 'input_queue') or not streaming.input_queue:
            streaming.input_queue = Queue()

    def execute(
        self, input_data: HumanFeedbackInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Execute the tool with the provided input data and configuration.

        This method prompts the user for input using the specified input method and returns the result.
        When auto_detect is enabled, it automatically chooses the best input method based on the environment.

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

        input_text = Template(self.msg_template).render(input_data.model_dump())

        if self.auto_detect:
            detected_method = self._detect_input_method(config)
            logger.debug(f"Tool {self.name} - {self.id}: auto-detected input method: {detected_method}")
            actual_method = detected_method
        else:
            actual_method = self.input_method

        if isinstance(actual_method, FeedbackMethod):
            if actual_method == FeedbackMethod.CONSOLE:
                result = self.input_method_console(input_text)
            elif actual_method == FeedbackMethod.STREAM:
                self._ensure_streaming_config(config)
                result = self.input_method_streaming(prompt=input_text, config=config, **kwargs)
            else:
                raise ValueError(f"Unsupported input method: {actual_method}")
        else:
            result = self.input_method.get_input(input_text)

        logger.debug(f"Tool {self.name} - {self.id}: finished with result {result}")
        return {"content": result}


class MessageSenderInputSchema(BaseModel):
    model_config = ConfigDict(extra="allow")


class MessageSenderTool(Node):
    """
    A tool for sending messages.

    Attributes:
        group (Literal[NodeGroup.TOOLS]): The group the node belongs to.
        name (str): The name of the tool.
        description (str): A brief description of the tool's purpose.
        msg_template (str): Template of message to send.
        output_method (FeedbackMethod | InputMethodCallable): The method used to gather user input.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "Message Sender Tool"
    description: str = """Sends messages to users through console or streaming output methods.

Key Capabilities:
- Message delivery via console or streaming interfaces
- Customizable message templates with parameter substitution
- Real-time status updates and notifications
- Integration with workflow callback systems

Usage Strategy:
- Send status updates during long-running processes
- Provide user notifications for completed operations
- Display error messages with context and details
- Broadcast information to connected clients

Parameter Guide:
- msg_template: Jinja2 template for message formatting
- Dynamic parameters: Content substituted into template

Examples:
- {"msg_template": "Process completed successfully"}
"""
    msg_template: str = "{{input}}"
    output_method: FeedbackMethod | OutputMethodCallable = FeedbackMethod.CONSOLE
    input_schema: ClassVar[type[MessageSenderInputSchema]] = MessageSenderInputSchema
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def update_description(self):
        msg_template = self.msg_template
        self.description += (
            f"\nThis is the template of message to send: '{msg_template}'."
            " Parameters will be substituted based on the provided input data."
        )
        return self

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
            source=StreamingEntitySource(
                name=self.name,
                group=self.group,
                type=self.type,
            ),
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
        input_text = Template(self.msg_template).render(input_data.model_dump())

        if isinstance(self.output_method, FeedbackMethod):
            if self.output_method == FeedbackMethod.CONSOLE:
                self.output_method_console(input_text)
            elif self.output_method == FeedbackMethod.STREAM:
                self.output_method_streaming(prompt=input_text, config=config, **kwargs)
            else:
                raise ValueError(f"Unsupported feedback method: {self.output_method}")
        else:
            self.output_method.send_message(input_text)

        logger.debug(f"Tool {self.name} - {self.id}: finished")
        return {"content": input_text}
