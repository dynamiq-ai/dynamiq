"""History and context management for agents."""

from typing import Callable

from dynamiq.nodes.agents.exceptions import ParsingError
from dynamiq.nodes.agents.utils import SummarizationConfig, XMLParser
from dynamiq.prompts import Message, MessageRole, Prompt, VisionMessage, VisionMessageTextContent
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger


class HistoryManager:
    """Manages conversation history, token limits, and summarization for agents."""

    def __init__(self):
        """Initialize the history manager without any agent dependency."""
        pass

    def is_token_limit_exceeded(
        self,
        prompt: Prompt,
        model: str,
        token_limit: int,
        summarization_config: SummarizationConfig,
    ) -> bool:
        """
        Check whether token limit for summarization is exceeded.

        Args:
            prompt: The prompt containing messages
            model: The model name for token counting
            token_limit: Maximum token limit for the model
            summarization_config: Summarization configuration

        Returns:
            bool: Whether token limit is exceeded
        """
        prompt_tokens = prompt.count_tokens(model)

        return (
            summarization_config.max_token_context_length
            and prompt_tokens > summarization_config.max_token_context_length
        ) or (prompt_tokens / token_limit > summarization_config.context_usage_ratio)

    def summarize_history(
        self,
        prompt: Prompt,
        input_message: Message | VisionMessage,
        summary_offset: int,
        history_offset: int,
        summarization_config: SummarizationConfig,
        history_prompt: str,
        max_attempts: int,
        run_llm: Callable,
        agent_name: str,
        agent_id: str,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> int:
        """
        Summarizes history and saves relevant information in the context.

        Args:
            prompt: The prompt containing messages to summarize
            input_message: User request message
            summary_offset: Offset to the position of the first message in prompt that was not summarized
            history_offset: Offset for the agent history
            summarization_config: Summarization configuration
            history_prompt: System prompt for history summarization
            max_attempts: Maximum number of attempts to generate summary
            run_llm: Function to call LLM (receives messages, config, **kwargs)
            agent_name: Agent name for logging
            agent_id: Agent ID for logging
            config: Configuration for the agent run
            **kwargs: Additional parameters for running the agent

        Returns:
            int: Number of summarized messages
        """
        logger.info(f"Agent {agent_name} - {agent_id}: Summarization of tool output started.")
        messages_history = "\nHistory to extract information from: \n"
        summary_sections = []

        offset = max(
            history_offset,
            summary_offset - summarization_config.context_history_length,
        )
        for index, message in enumerate(prompt.messages[offset:]):
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

        summary_messages = [
            Message(content=history_prompt, role=MessageRole.SYSTEM, static=True),
            input_message,
            Message(content=messages_history, role=MessageRole.USER, static=True),
        ]

        summary_tags = [f"tool_output{index}" for index in summary_sections]
        for _ in range(max_attempts):
            llm_result = run_llm(
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
                prompt.messages[message_index].content = (
                    f"Observation (shortened): \n{parsed_data.get(summary_tags[summary_index])}"
                )

            # Check if still exceeded after shortening
            model = kwargs.get("model", "")
            token_limit = kwargs.get("token_limit", float("inf"))
            if self.is_token_limit_exceeded(prompt, model, token_limit, summarization_config):
                prompt.messages[summary_sections[-1]].content = (
                    f"Observation (shortened): \n{parsed_data.get(summary_tags[-1])}"
                )
                summary_offset = len(prompt.messages)
            else:
                summary_offset = len(prompt.messages) - 2

            logger.info(f"Agent {agent_name} - {agent_id}: Summarization of tool output finished.")
            return summary_offset

        # If we exhausted all attempts without success, return current offset
        logger.warning(f"Agent {agent_name} - {agent_id}: Failed to summarize after {max_attempts} attempts.")
        return summary_offset

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
