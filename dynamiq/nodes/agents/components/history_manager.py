"""History and context management for agents."""

from typing import TYPE_CHECKING

from dynamiq.nodes.agents.exceptions import ParsingError
from dynamiq.nodes.agents.utils import XMLParser
from dynamiq.prompts import Message, MessageRole, VisionMessage, VisionMessageTextContent
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

if TYPE_CHECKING:
    from dynamiq.nodes.agents.agent import Agent


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
        Summarizes history and saves relevant information in the context.

        Args:
            input_message: User request message
            summary_offset: Offset to the position of the first message in prompt that was not summarized
            config: Configuration for the agent run
            **kwargs: Additional parameters for running the agent

        Returns:
            int: Number of summarized messages
        """
        logger.info(f"Agent {self.agent.name} - {self.agent.id}: Summarization of tool output started.")
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
        for _ in range(self.agent.max_loops):
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

            logger.info(f"Agent {self.agent.name} - {self.agent.id}: Summarization of tool output finished.")
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
