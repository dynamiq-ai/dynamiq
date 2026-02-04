from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, ClassVar, Literal

from jinja2 import Template
from pydantic import BaseModel, ConfigDict, Field, model_validator

from dynamiq.nodes import NodeGroup
from dynamiq.nodes.node import Node, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.types.feedback import FeedbackMethod
from dynamiq.types.streaming import StreamingEntitySource, StreamingEventMessage
from dynamiq.utils.logger import logger


class HumanFeedbackAction(str, Enum):
    """Actions available for the HumanFeedbackTool."""

    ASK = "ask"  # Request input from user
    INFO = "info"  # Send info message without waiting for response


class HFStreamingInputEventMessageData(BaseModel):
    content: str


class HFStreamingInputEventMessage(StreamingEventMessage):
    data: HFStreamingInputEventMessageData


class HFStreamingOutputEventMessageData(BaseModel):
    prompt: str
    action: HumanFeedbackAction = HumanFeedbackAction.ASK


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
    to send messages in the HumanFeedbackTool (action='info').
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
    """Input schema for HumanFeedbackTool."""

    action: HumanFeedbackAction = Field(
        default=HumanFeedbackAction.ASK,
        description="Action to perform: 'ask' to request input from user, 'info' to just send a message.",
    )
    model_config = ConfigDict(extra="allow")


class HumanFeedbackTool(Node):
    """
    A unified tool for human interaction - both gathering feedback and sending messages.

    This tool can either prompt the user for input (action="ask") or send an info message
    without waiting for response (action="info").

    Attributes:
        group (Literal[NodeGroup.TOOLS]): The group the node belongs to.
        name (str): The name of the tool.
        description (str): A brief description of the tool's purpose.
        msg_template (str): Template of message to send.
        input_method (FeedbackMethod | InputMethodCallable): The method used to gather user input.
        output_method (FeedbackMethod | OutputMethodCallable): The method used to send messages.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "message_sender"
    description: str = """A tool for gathering approval, confirmation, clarification, or information from user and
  sending status updates.

Use 'ask' action to request input - workflow WAITS for user response before continuing.
Use 'info' action to send notification - workflow continues immediately without waiting.

Examples:
- {"action": "ask", "input": "Should I proceed? (yes/no)"}
- {"action": "info", "input": "Task completed."}

Important:
- Use 'ask' for approval, confirmation, clarification, or information.
- The user can only provide text responses - they can not perform actions.
- This tool should be used to gather information from user and send status updates during agent execution.
"""
    input_method: FeedbackMethod | InputMethodCallable = FeedbackMethod.CONSOLE
    output_method: FeedbackMethod | OutputMethodCallable = FeedbackMethod.CONSOLE
    action: HumanFeedbackAction | None = Field(
        default=None,
        description="If set, this action is always used, ignoring input. Useful for workflow nodes.",
    )
    input_schema: ClassVar[type[HumanFeedbackInputSchema]] = HumanFeedbackInputSchema
    msg_template: str = "{{input}}"
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def update_description(self):
        msg_template = self.msg_template
        self.description += (
            f"\nMessage template: '{msg_template}'." " Parameters will be substituted based on the provided input data."
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
            data=HFStreamingOutputEventMessageData(prompt=prompt, action=HumanFeedbackAction.ASK),
            event=streaming.event,
            source=StreamingEntitySource(
                id=self.id,
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

    def output_method_console(self, message: str) -> None:
        """
        Sends message to console.

        Args:
            message (str): The message to display to the user.
        """
        print(message)

    def output_method_streaming(self, message: str, config: RunnableConfig, **kwargs) -> None:
        """
        Sends message using streaming method.

        Args:
            message (str): The message to display to the user.
            config (RunnableConfig, optional): The configuration for the runnable. Defaults to None.
        """
        streaming = getattr(config.nodes_override.get(self.id), "streaming", None) or self.streaming

        event = HFStreamingOutputEventMessage(
            wf_run_id=config.run_id,
            entity_id=self.id,
            data=HFStreamingOutputEventMessageData(prompt=message, action=HumanFeedbackAction.INFO),
            event=streaming.event,
            source=StreamingEntitySource(
                id=self.id,
                name=self.name,
                group=self.group,
                type=self.type,
            ),
        )
        logger.debug(f"Tool {self.name} - {self.id}: sending output event {event}")
        self.run_on_node_execute_stream(callbacks=config.callbacks, event=event, **kwargs)

    def _execute_ask(self, input_text: str, config: RunnableConfig, **kwargs) -> str:
        """Execute the 'ask' action - get input from user."""
        if isinstance(self.input_method, FeedbackMethod):
            if self.input_method == FeedbackMethod.CONSOLE:
                return self.input_method_console(input_text)
            elif self.input_method == FeedbackMethod.STREAM:
                streaming = getattr(config.nodes_override.get(self.id), "streaming", None) or self.streaming
                if not streaming.input_streaming_enabled:
                    raise ValueError(
                        f"'{FeedbackMethod.STREAM}' input method requires enabled input and output streaming."
                    )
                return self.input_method_streaming(prompt=input_text, config=config, **kwargs)
            else:
                raise ValueError(f"Unsupported input method: {self.input_method}")
        else:
            return self.input_method.get_input(input_text)

    def _execute_send(self, input_text: str, config: RunnableConfig, **kwargs) -> None:
        """Execute the 'info' action - send info message to user."""
        if isinstance(self.output_method, FeedbackMethod):
            if self.output_method == FeedbackMethod.CONSOLE:
                self.output_method_console(input_text)
            elif self.output_method == FeedbackMethod.STREAM:
                self.output_method_streaming(message=input_text, config=config, **kwargs)
            else:
                raise ValueError(f"Unsupported output method: {self.output_method}")
        else:
            self.output_method.send_message(input_text)

    def execute(
        self, input_data: HumanFeedbackInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Execute the tool with the provided input data and configuration.

        Based on the 'action' parameter:
        - "ask": Prompts the user for input and returns their response
        - "info": Sends an info message to the user without waiting for response

        Args:
            input_data (HumanFeedbackInputSchema): The input data containing action and message.
            config (RunnableConfig, optional): The configuration for the runnable. Defaults to None.
            **kwargs: Additional keyword arguments to be passed to the node execute run.

        Returns:
            dict[str, Any]: A dictionary containing the result under the 'content' key.
        """
        logger.debug(f"Tool {self.name} - {self.id}: started with input data {input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        input_text = Template(self.msg_template).render(input_data.model_dump())
        action = self.action if self.action is not None else input_data.action

        if action == HumanFeedbackAction.ASK:
            result = self._execute_ask(input_text, config, **kwargs)
            logger.debug(f"Tool {self.name} - {self.id}: finished with result {result}")
            return {"content": result}
        elif action == HumanFeedbackAction.INFO:
            self._execute_send(input_text, config, **kwargs)
            logger.debug(f"Tool {self.name} - {self.id}: message sent")
            return {"content": f"Message sent: {input_text}"}
        else:
            raise ValueError(
                f"Unsupported action: {action}. Use '{HumanFeedbackAction.ASK}' or '{HumanFeedbackAction.INFO}'."
            )
