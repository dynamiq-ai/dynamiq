from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, ClassVar, Literal

from jinja2 import Template
from pydantic import BaseModel, ConfigDict, Field, model_validator

from dynamiq.nodes import NodeGroup
from dynamiq.nodes.node import Node, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.types.cancellation import check_cancellation
from dynamiq.types.feedback import FeedbackMethod
from dynamiq.types.streaming import StreamingEntitySource, StreamingEventMessage
from dynamiq.utils import generate_uuid
from dynamiq.utils.logger import logger


class HumanFeedbackAction(str, Enum):
    """Actions available for the HumanFeedbackTool."""

    ASK = "ask"  # Request input from user
    INFO = "info"  # Send info message without waiting for response


class QuestionOption(BaseModel):
    """A single choice offered for a structured question."""

    label: str
    description: str | None = None


class Question(BaseModel):
    """A single structured question, one of up to 4 asked in a single 'ask' batch."""

    id: str | None = Field(default=None, description="Defaults to the question's index within the batch.")
    header: str | None = Field(default=None, description="Short chip label for the question, e.g. 'Scope'.")
    question: str
    options: list[QuestionOption] = Field(default_factory=list)
    multi_select: bool = False
    allow_custom_answer: bool = Field(
        default=True, description="Whether a free-text 'Other' answer is allowed alongside/instead of options."
    )

    @model_validator(mode="after")
    def validate_options(self):
        if self.options and not (2 <= len(self.options) <= 4):
            raise ValueError("'options' must contain between 2 and 4 choices when provided.")
        return self


class Answer(BaseModel):
    """A user's answer to one Question."""

    question_id: str
    selected: list[str] = Field(default_factory=list)
    custom_text: str | None = None


class HFStreamingInputEventMessageData(BaseModel):
    content: str = ""
    answers: list[Answer] | None = None
    request_id: str | None = None


class HFStreamingInputEventMessage(StreamingEventMessage):
    data: HFStreamingInputEventMessageData


class HFStreamingOutputEventMessageData(BaseModel):
    prompt: str
    action: HumanFeedbackAction = HumanFeedbackAction.ASK
    is_browser_takeover: bool = False
    questions: list[Question] | None = None
    request_id: str | None = None


class HFStreamingOutputEventMessage(StreamingEventMessage):
    data: HFStreamingOutputEventMessageData


def _render_questions_prompt(questions: list[Question]) -> str:
    """Render structured questions as numbered text, for text-only consumers (Slack, old UIs, LLM fallback)."""
    lines = []
    for index, question in enumerate(questions, start=1):
        header = f"[{question.header}] " if question.header else ""
        lines.append(f"{index}. {header}{question.question}")
        for option in question.options:
            suffix = f" - {option.description}" if option.description else ""
            lines.append(f"   - {option.label}{suffix}")
        if question.allow_custom_answer:
            lines.append("   - Other (free text)")
    return "\n".join(lines)


def _format_answers_for_llm(questions: list[Question], answers: list[Answer]) -> str:
    """Format structured answers as a readable observation for the LLM."""
    questions_by_id = {(question.id or str(index)): question for index, question in enumerate(questions)}
    lines = []
    for answer in answers:
        question = questions_by_id.get(answer.question_id)
        label = question.question if question else answer.question_id
        parts = list(answer.selected)
        if answer.custom_text:
            parts.append(f"other: {answer.custom_text}")
        lines.append(f"Q: {label} -> A: {', '.join(parts) if parts else '(no answer)'}")
    return "\n".join(lines)


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
    input: str = Field(
        default="",
        description="The message or question shown to the user. Rendered via the message template "
        "(default template is '{{input}}').",
    )
    questions: list[Question] | None = Field(
        default=None,
        description="Optional, for action='ask' only: 1-4 structured questions, each offering 2-4 selectable "
        "options (single- or multi-select) plus an optional free-text 'Other' answer.",
    )
    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def validate_questions(self):
        if self.questions is not None and not (1 <= len(self.questions) <= 4):
            raise ValueError("'questions' must contain between 1 and 4 questions.")
        return self


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
    name: str = "message-sender"
    description: str = """A tool for gathering approval, confirmation, clarification, or information from user and
  sending status updates.

Use 'ask' action to request input - workflow WAITS for user response before continuing.
Use 'info' action to send notification - workflow continues immediately without waiting.
For 'ask', you may optionally pass 1-4 structured 'questions' instead of (or in addition to) free-text
'input' - each question can offer 2-4 selectable options (single- or multi-select) plus a free-text
'Other' answer, so the user can click instead of typing.

Examples:
- {"action": "ask", "input": "Should I proceed? (yes/no)"}
- {"action": "ask", "questions": [{"question": "Which scope?", "options": [{"label": "Q1 only"}, \
{"label": "Full year"}]}]}
- {"action": "info", "input": "Task completed."}

Important:
- Use 'ask' for approval, confirmation, clarification, or information.
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
    is_browser_takeover: bool = Field(
        default=False,
        description="If True, streamed feedback events are marked as a browser-takeover request so a chat UI can "
        "render an interactive browser session instead of a plain text prompt. Requires a browser tool "
        "(e.g. Stagehand with live view enabled) in the same run to provide the live session.",
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)

    _GUIDANCE_ANCHOR: ClassVar[str] = "\nInteraction mode:"

    @model_validator(mode="after")
    def update_description(self):
        # Strip any previously generated guidance before regenerating so repeated validation,
        # serialization round-trips, and flag changes stay idempotent and never stack
        # conflicting notes (e.g. both the takeover and the text-only line).
        base = self.description.split(self._GUIDANCE_ANCHOR, 1)[0].rstrip()

        if self.is_browser_takeover:
            interaction = (
                "Interaction mode: browser takeover is ENABLED - the user interacts directly with a live "
                "browser session (navigating, clicking, typing), not only text responses. Coordinate with the "
                "browser tool and hand off control to the user when manual action is required."
            )
        else:
            interaction = "Interaction mode: the user can only provide text responses - they can not perform actions."

        self.description = (
            f"{base}\n{interaction}"
            f"\nMessage template: '{self.msg_template}'."
            " Parameters will be substituted based on the provided input data."
        )
        return self

    def input_method_console(self, prompt: str, config: RunnableConfig = None) -> str:
        """
        Get input from the user using the console input method.
        Cancellable: runs input() in a daemon thread and polls for cancellation.

        Args:
            prompt (str): The prompt to display to the user.
            config (RunnableConfig, optional): Configuration for cancellation check.

        Returns:
            str: The user's input.
        """
        import threading as _threading

        check_cancellation(config)

        result = {}

        def _read_input():
            result["feedback"] = input(prompt)

        input_thread = _threading.Thread(target=_read_input, daemon=True)
        input_thread.start()

        while input_thread.is_alive():
            check_cancellation(config)
            input_thread.join(timeout=0.5)

        return result.get("feedback", "")

    def input_method_streaming(
        self, prompt: str, config: RunnableConfig, questions: list[Question] | None = None, **kwargs
    ) -> tuple[str, list[Answer] | None]:
        """
        Get input from the user using the queue streaming input method.

        Args:
            prompt (str): The prompt to display to the user.
            config (RunnableConfig, optional): The configuration for the runnable. Defaults to None.
            questions (list[Question] | None): Optional structured questions to ask alongside/instead of prompt.

        Returns:
            tuple[str, list[Answer] | None]: The rendered text observation, and the raw structured
                answers if the reply carried any.
        """
        logger.debug(f"Tool {self.name} - {self.id}: started with prompt {prompt}")
        check_cancellation(config)

        streaming = getattr(config.nodes_override.get(self.id), "streaming", None) or self.streaming

        # Minted per ask so a reply can be matched to its round; also lets the queue reader
        # discard a stale answer left over from an earlier question (see get_input_streaming_event).
        request_id = generate_uuid()
        if questions:
            for index, question in enumerate(questions):
                if question.id is None:
                    question.id = str(index)

        event = HFStreamingOutputEventMessage(
            wf_run_id=config.run_id,
            entity_id=self.id,
            data=HFStreamingOutputEventMessageData(
                prompt=prompt,
                action=HumanFeedbackAction.ASK,
                is_browser_takeover=self.is_browser_takeover,
                questions=questions,
                request_id=request_id,
            ),
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
            request_id=request_id,
        )
        logger.debug(f"Tool {self.name} - {self.id}: received input event {event}")

        answers = event.data.answers
        content = event.data.content
        if answers and not content:
            content = _format_answers_for_llm(questions or [], answers)
        return content, answers

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
            data=HFStreamingOutputEventMessageData(
                prompt=message, action=HumanFeedbackAction.INFO, is_browser_takeover=self.is_browser_takeover
            ),
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

    def _execute_ask(
        self, input_text: str, config: RunnableConfig, questions: list[Question] | None = None, **kwargs
    ) -> tuple[str, list[Answer] | None]:
        """Execute the 'ask' action - get input from user."""
        check_cancellation(config)
        if isinstance(self.input_method, FeedbackMethod):
            if self.input_method == FeedbackMethod.CONSOLE:
                return self.input_method_console(input_text, config=config), None
            elif self.input_method == FeedbackMethod.STREAM:
                streaming = getattr(config.nodes_override.get(self.id), "streaming", None) or self.streaming
                if not streaming.input_streaming_enabled:
                    raise ValueError(
                        f"'{FeedbackMethod.STREAM}' input method requires enabled input and output streaming."
                    )
                return self.input_method_streaming(prompt=input_text, config=config, questions=questions, **kwargs)
            else:
                raise ValueError(f"Unsupported input method: {self.input_method}")
        else:
            return self.input_method.get_input(input_text), None

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
            dict[str, Any]: A dictionary with the result under 'content', plus 'answers' (the raw
                structured answers) when the reply to a structured ask carried any.
        """
        logger.debug(f"Tool {self.name} - {self.id}: started with input data {input_data.model_dump()}")
        config = ensure_config(config)
        check_cancellation(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        input_text = Template(self.msg_template).render(input_data.model_dump())
        action = self.action if self.action is not None else input_data.action

        if action == HumanFeedbackAction.ASK:
            # Browser takeover hands control to the user directly; structured click-through
            # questions don't apply there, so any questions are ignored in that mode.
            questions = getattr(input_data, "questions", None) if not self.is_browser_takeover else None
            prompt_text = input_text
            if questions:
                rendered_questions = _render_questions_prompt(questions)
                prompt_text = f"{input_text}\n{rendered_questions}".strip() if input_text else rendered_questions

            content, answers = self._execute_ask(prompt_text, config, questions=questions, **kwargs)
            logger.debug(f"Tool {self.name} - {self.id}: finished with result {content}")
            result = {"content": content}
            if answers is not None:
                result["answers"] = [answer.model_dump() for answer in answers]
            return result
        elif action == HumanFeedbackAction.INFO:
            self._execute_send(input_text, config, **kwargs)
            logger.debug(f"Tool {self.name} - {self.id}: message sent")
            return {"content": f"Message sent: {input_text}"}
        else:
            raise ValueError(
                f"Unsupported action: {action}. Use '{HumanFeedbackAction.ASK}' or '{HumanFeedbackAction.INFO}'."
            )
