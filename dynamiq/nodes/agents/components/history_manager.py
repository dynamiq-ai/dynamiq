"""History and context management mixin for agents."""

from dynamiq.prompts import Message, MessageRole, VisionMessage, VisionMessageTextContent
from dynamiq.utils.logger import logger


class HistoryManagerMixin:
    """Mixin that provides conversation history and token limit management.

    This mixin expects the class using it to have the following attributes:
    - _prompt: Prompt object with messages
    - llm: LLM object with model and get_token_limit()
    - summarization_config: SummarizationConfig object
    - _history_offset: int offset past the system prompt (typically 1);
      everything from this index onward is eligible for summarization.
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

    def _split_history(
        self,
    ) -> tuple[list[Message | VisionMessage], list[Message | VisionMessage]]:
        """Split conversation history into messages to summarize and messages to preserve.

        Uses ``preserve_last_messages`` from summarization config to determine
        the split point.  When the history is too short to split, all messages
        go to the summarize bucket and nothing is preserved.

        Returns:
            Tuple of (to_summarize, to_preserve).
        """
        preserve_count = self.summarization_config.preserve_last_messages
        conversation_history = self._prompt.messages[self._history_offset :]

        if preserve_count > 0 and len(conversation_history) > preserve_count:
            to_summarize = conversation_history[:-preserve_count]
            to_preserve = [m.copy() for m in conversation_history[-preserve_count:]]
        else:
            to_summarize = conversation_history
            to_preserve = []

        return to_summarize, to_preserve

    def _compact_history(
        self,
        summary: str | None = None,
        pinned_content: str | None = None,
    ) -> None:
        """Compact history, optionally inserting a summary before preserved messages.

        Replaces the conversation history with::

            [system prefix] [summary + original request] [preserved msgs …]

        If *pinned_content* is provided and not present in the preserved tail,
        the original user request is appended verbatim to the summary so it is
        never lost across repeated compactions.

        Args:
            summary: Optional summary text to insert after prefix.
            pinned_content: Plain-text content of the original user request.
        """
        _, preserved = self._split_history()

        self._prompt.messages = self._prompt.messages[: self._history_offset]

        if summary:
            if pinned_content is not None:
                preserved_has_pinned = any(self._extract_message_text(m) == pinned_content for m in preserved)
                if not preserved_has_pinned:
                    summary = f"{summary}\n\nOriginal request: {pinned_content}"

            self._prompt.messages.append(
                Message(role=MessageRole.USER, content=f"\nObservation: {summary}\n", static=True)
            )

        self._prompt.messages.extend(preserved)
        logger.info(
            f"Agent {self.name} - {self.id}: History compacted. "
            f"Summary: {'yes' if summary else 'no'}, preserved: {len(preserved)} messages."
        )

    @staticmethod
    def _extract_message_text(message: Message | VisionMessage) -> str:
        """Extract plain text from a Message or VisionMessage."""
        if isinstance(message, Message):
            return message.content
        text_parts = [c.text for c in message.content if isinstance(c, VisionMessageTextContent)]
        return " ".join(text_parts) if text_parts else "Image input"

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
