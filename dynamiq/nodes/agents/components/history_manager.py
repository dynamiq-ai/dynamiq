"""History and context management for agents."""

from typing import TYPE_CHECKING

from dynamiq.nodes.agents.exceptions import ParsingError
from dynamiq.nodes.agents.utils import SummarizationMode, XMLParser
from dynamiq.prompts import Message, MessageRole, VisionMessage, VisionMessageTextContent
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

if TYPE_CHECKING:
    from dynamiq.nodes.agents.agent import Agent


# Default summary request prompt
DEFAULT_SUMMARY_PROMPT = (
    "Please provide a concise summary of the conversation history. "
    "Focus on key decisions, important information, and tool outputs. "
    "Wrap your summary in <summary></summary> tags."
)


class HistoryManager:
    """Manages conversation history, token limits, and summarization for agents."""

    def __init__(self, agent: "Agent"):
        """
        Initialize the history manager.

        Args:
            agent: The agent instance to manage history for
        """
        self.agent = agent

    def is_token_limit_exceeded(self) -> bool:
        """
        Check whether token limit for summarization is exceeded.

        Returns:
            bool: Whether token limit is exceeded
        """
        prompt_tokens = self.agent._prompt.count_tokens(self.agent.llm.model)

        return (
            self.agent.summarization_config.max_token_context_length
            and prompt_tokens > self.agent.summarization_config.max_token_context_length
        ) or (prompt_tokens / self.agent.llm.get_token_limit() > self.agent.summarization_config.context_usage_ratio)

    def summarize_history(
        self,
        input_message: Message | VisionMessage,
        summary_offset: int,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> int:
        """
        Summarizes history based on the configured mode.

        Modes:
        - SummarizationMode.REPLACE: Replace entire message history with summary (default, memory efficient)
        - SummarizationMode.PRESERVE: Preserve message structure, only shorten tool outputs (legacy)

        Args:
            input_message: User request message
            summary_offset: Offset to the position of the first message in prompt that was not summarized
            config: Configuration for the agent run
            **kwargs: Additional parameters for running the agent

        Returns:
            int: Number of messages in the new history (after summarization) or updated offset
        """

        if self.agent.summarization_config.mode == SummarizationMode.PRESERVE:
            return self._summarize_preserve_structure(input_message, summary_offset, config, **kwargs)
        else:
            return self._summarize_replace_history(input_message, summary_offset, config, **kwargs)

    # -------------------------------- Generate summary for replace mode ------------------------------------- #

    def _generate_summary(
        self,
        input_message: Message | VisionMessage,
        summary_offset: int,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> str:
        """
        Generate a structured summary of the conversation history.

        This is a separate function that injects a summary prompt as a user turn,
        and the LLM generates a structured summary wrapped in <summary></summary> tags.

        Args:
            input_message: User request message
            summary_offset: Starting index for messages to include in summarization
            config: Configuration for the agent run
            **kwargs: Additional parameters for running the agent

        Returns:
            str: The extracted summary content
        """
        logger.info(f"Agent {self.agent.name} - {self.agent.id}: Generating conversation summary.")

        # Get conversation history messages to be summarized (starting from summary_offset)
        conversation_history_messages = self._get_history_messages_for_summarization(summary_offset)

        # Get model-specific system prompt for history summarization
        system_prompt_for_summarization = self.agent.system_prompt_manager.history_prompt

        summary_messages = (
            [
                Message(content=system_prompt_for_summarization, role=MessageRole.SYSTEM, static=True),
                input_message,
            ]
            + conversation_history_messages
            + [
                Message(
                    content=DEFAULT_SUMMARY_PROMPT,
                    role=MessageRole.USER,
                    static=True,
                ),
            ]
        )

        for attempt in range(self.agent.summarization_config.max_attempts):
            llm_result = self.agent._run_llm(
                messages=summary_messages,
                config=config,
                **kwargs,
            )

            output = llm_result.output["content"]

            try:
                summary = XMLParser.extract_first_tag_regex(output, ["summary"])

                if summary:
                    logger.info(f"Agent {self.agent.name} - {self.agent.id}: Summary generated successfully.")
                    return summary
                else:
                    parsed_data = XMLParser.parse(
                        f"<root>{output}</root>",
                        required_tags=["summary"],
                        optional_tags=[],
                    )
                    summary = parsed_data.get("summary", "")
                    if summary:
                        logger.info(f"Agent {self.agent.name} - {self.agent.id}: Summary generated successfully.")
                        return summary

            except ParsingError as e:
                logger.warning(
                    f"Agent {self.agent.name} - {self.agent.id}: "
                    f"Failed to extract summary on attempt {attempt + 1}: {e}"
                )
                summary_messages.append(Message(content=output, role=MessageRole.ASSISTANT, static=True))
                summary_messages.append(
                    Message(
                        content=(
                            f"Error: {e}. Please provide the summary wrapped in <summary></summary> tags. "
                            "Ensure the tags are properly formatted."
                        ),
                        role=MessageRole.USER,
                        static=True,
                    )
                )
                continue

        logger.warning(
            f"Agent {self.agent.name} - {self.agent.id}: "
            f"Could not extract summary after 3 attempts. Using raw output."
        )
        return output

    def _summarize_replace_history(
        self,
        input_message: Message | VisionMessage,
        summary_offset: int,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> int:
        """
        Replace entire message history with summary (default mode).

        Implementation approach:
        1. Summary generation: When threshold is exceeded, inject a summary prompt
           and generate a structured summary wrapped in <summary></summary> tags
        2. Context replacement: Extract the summary and replace entire message history with it
        3. Continuation: The conversation resumes from the summary

        Args:
            input_message: User request message
            summary_offset: Offset to the position of the first message in prompt that was not summarized
            config: Configuration for the agent run
            **kwargs: Additional parameters for running the agent

        Returns:
            int: Number of messages in the new history (after summarization)
        """
        logger.info(f"Agent {self.agent.name} - {self.agent.id}: Starting history summarization (replace mode).")

        # Step 1: Generate summary (separate function, separate from main loop)
        summary = self._generate_summary(input_message, summary_offset, config, **kwargs)

        # Step 2: Context replacement - replace entire message history with the summary
        # Keep only static system messages and replace the rest with the summary
        static_system_messages = [
            msg for msg in self.agent._prompt.messages if msg.role == MessageRole.SYSTEM and msg.static
        ]

        # Create new history with system messages + summary
        self.agent._prompt.messages = static_system_messages + [
            Message(
                content=f"Conversation Summary:\n{summary}",
                role=MessageRole.USER,
                static=False,
            )
        ]

        logger.info(
            f"Agent {self.agent.name} - {self.agent.id}: "
            f"History replaced with summary. New message count: {len(self.agent._prompt.messages)}"
        )

        # Step 3: Continuation - return the new offset
        # The conversation will resume from this point
        return len(self.agent._prompt.messages)

    # -------------------------------- Generate summary for replace mode ------------------------------------- #

    # -------------------------------- Generate summary for preserve mode ------------------------------------- #

    def _summarize_preserve_structure(
        self,
        input_message: Message | VisionMessage,
        summary_offset: int,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> int:
        """
        Preserve message structure and shorten tool outputs (legacy mode).

        This implementation summarizes individual tool outputs
        rather than replacing the entire history.

        Args:
            input_message: User request message
            summary_offset: Offset to the position of the first message in prompt that was not summarized
            config: Configuration for the agent run
            **kwargs: Additional parameters for running the agent

        Returns:
            int: Updated summary offset
        """
        logger.info(f"Agent {self.agent.name} - {self.agent.id}: Summarization of tool output started (preserve mode).")
        messages_history = "\nHistory to extract information from: \n"
        summary_sections = []

        offset = max(
            self.agent._history_offset,
            summary_offset - self.agent.summarization_config.context_history_length,
        )
        for index, message in enumerate(self.agent._prompt.messages[offset:]):
            if message.role == MessageRole.USER:
                if (index + offset >= summary_offset) and ("Observation:" in message.content):
                    messages_history += (
                        f"=== TOOL_OUTPUT: {index + offset} === \n {message.content}"
                        f"\n === TOOL_OUTPUT: {index + offset} === \n"
                    )
                    summary_sections.append(index + offset)
            else:
                messages_history += f"\n{message.content}\n"

        messages_history = (
            messages_history + f"\n Required tags in the output {[f'tool_output{index}' for index in summary_sections]}"
        )

        # Use model-specific history summarization prompt from prompt manager
        history_prompt = self.agent.system_prompt_manager.history_prompt

        summary_messages = [
            Message(content=history_prompt, role=MessageRole.SYSTEM, static=True),
            input_message,
            Message(content=messages_history, role=MessageRole.USER, static=True),
        ]

        summary_tags = [f"tool_output{index}" for index in summary_sections]
        for _ in range(self.agent.summarization_config.max_attempts):
            llm_result = self.agent._run_llm(
                messages=summary_messages,
                config=config,
                **kwargs,
            )

            output = llm_result.output["content"]

            summary_messages.append(Message(content=output, role=MessageRole.ASSISTANT, static=True))
            try:
                parsed_data = XMLParser.parse(
                    f"<root>{output}</root>",
                    required_tags=summary_tags,
                    optional_tags=[],
                )
            except ParsingError as e:
                logger.error(f"Error: {e}. Make sure you have provided all tags at once: {summary_tags}")
                summary_messages.append(Message(content=str(e), role=MessageRole.USER, static=True))
                continue

            for summary_index, message_index in enumerate(summary_sections[:-1]):
                self.agent._prompt.messages[message_index].content = (
                    f"Observation (shortened): \n{parsed_data.get(summary_tags[summary_index])}"
                )

            if self.is_token_limit_exceeded():
                self.agent._prompt.messages[summary_sections[-1]].content = (
                    f"Observation (shortened): \n{parsed_data.get(summary_tags[-1])}"
                )
                summary_offset = len(self.agent._prompt.messages)
            else:
                summary_offset = len(self.agent._prompt.messages) - 2

            logger.info(
                f"Agent {self.agent.name} - {self.agent.id}: Summarization of tool output finished (preserve mode)."
            )
            return summary_offset

        logger.warning(
            f"Agent {self.agent.name} - {self.agent.id}: "
            f"Preserve mode summarization failed after {self.agent.summarization_config.max_attempts} attempts."
        )
        return summary_offset

    # -------------------------------- Generate summary for preserve mode ------------------------------------- #

    def _get_history_messages_for_summarization(self, summary_offset: int) -> list[Message | VisionMessage]:
        """
        Get conversation history messages as a list starting from summary_offset.

        Args:
            summary_offset: Starting index for messages to include in summarization

        Returns:
            list[Message | VisionMessage]: List of history messages starting from summary_offset
        """
        return self.agent._prompt.messages[summary_offset:]

    def _format_history_for_summarization(self) -> str:
        """
        Format the conversation history for summarization.

        Returns:
            str: Formatted history content
        """
        history = []

        for message in self.agent._prompt.messages:
            if message.static:
                # Skip static system messages
                continue

            if isinstance(message, VisionMessage):
                for content in message.content:
                    if isinstance(content, VisionMessageTextContent):
                        history.append(f"[{message.role.value.upper()}]: {content.text}")
            else:
                history.append(f"[{message.role.value.upper()}]: {message.content}")

        return "\n\n".join(history)

    @staticmethod
    def aggregate_history(messages: list[Message | VisionMessage]) -> str:
        """
        Concatenates multiple history messages into one unified string.

        Args:
            messages: List of messages to aggregate

        Returns:
            str: Aggregated content
        """
        history = ""

        for message in messages:
            if isinstance(message, VisionMessage):
                for content in message.content:
                    if isinstance(content, VisionMessageTextContent):
                        history += content.text
            else:
                if message.role == MessageRole.ASSISTANT:
                    history += f"-TOOL DESCRIPTION START-\n{message.content}\n-TOOL DESCRIPTION END-\n"
                elif message.role == MessageRole.USER:
                    history += f"-TOOL OUTPUT START-\n{message.content}\n-TOOL OUTPUT END-\n"

        return history
