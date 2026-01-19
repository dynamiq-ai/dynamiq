"""History and context management mixin for agents."""

from dynamiq.prompts import Message, MessageRole, VisionMessage, VisionMessageTextContent
from dynamiq.utils.logger import logger


class HistoryManagerMixin:
    """Mixin that provides conversation history and token limit management.

    This mixin expects the class using it to have the following attributes:
    - _prompt: Prompt object with messages
    - llm: LLM object with model and get_token_limit()
    - summarization_config: SummarizationConfig object
    - _history_offset: int offset for history
    - name: str agent name
    - id: str agent id
    """

    def is_token_limit_exceeded(self) -> bool:
        """
        Check whether token limit for summarization is exceeded.

        Returns:
            bool: Whether token limit is exceeded
        """
        prompt_tokens = self._prompt.count_tokens(self.llm.model)

        return (
            self.summarization_config.max_token_context_length
            and prompt_tokens > self.summarization_config.max_token_context_length
        ) or (prompt_tokens / self.llm.get_token_limit() > self.summarization_config.context_usage_ratio)

    def _apply_context_manager_effect(self) -> None:
        """
        Apply context cleaning effect - replaces history with summary.

        Args:
            summary: The summarization result
        """
        try:

            logger.info(
                f"Agent {self.name} - {self.id}: Context Manager Tool completed. " f"Applying context cleaning effect."
            )

            # Keep messages up to history offset (system messages, initial user message)
            new_messages = self._prompt.messages[: self._history_offset]

            # Append copy of the last message before cleanup
            if new_messages and len(self._prompt.messages) > self._history_offset:
                new_messages.append(self._prompt.messages[-1].copy())

            # Replace the entire message list
            self._prompt.messages = new_messages

            logger.info(
                f"Agent {self.name} - {self.id}: Context cleaned. "
                f"Kept {self._history_offset} prefix messages. New count: {len(self._prompt.messages)}"
            )

        except Exception as e:
            logger.error(f"Agent {self.name} - {self.id}: Error applying context manager effect: {e}")
            raise e

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
