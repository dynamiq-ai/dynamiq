"""History and context management mixin for agents."""

from litellm import token_counter

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

    _running_summary: str | None = None

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

    def _compact_history(self, summary: str | None = None) -> None:
        """Compact history, optionally inserting a summary before preserved messages.

        Replaces the conversation history with::

            [prefix] [summary] [preserved msg 1] [preserved msg 2]

        The last ``preserve_last_messages`` messages are kept verbatim.
        If *summary* is provided it is inserted as a user message between the
        prefix and the preserved messages.  When incremental summarization is
        enabled, the summary is also stored as ``_running_summary``.

        Args:
            summary: Optional summary text to insert after prefix.
        """
        preserve_n = self.summarization_config.preserve_last_messages
        all_history = self._prompt.messages[self._history_offset :]

        if preserve_n > 0 and len(all_history) >= preserve_n:
            preserved = [m.copy() for m in all_history[-preserve_n:]]
        else:
            preserved = [m.copy() for m in all_history]

        self._prompt.messages = self._prompt.messages[: self._history_offset]

        if summary:
            if self.summarization_config.incremental:
                self._running_summary = summary
            self._prompt.messages.append(
                Message(role=MessageRole.USER, content=f"\nObservation: {summary}\n", static=True)
            )

        self._prompt.messages.extend(preserved)
        logger.info(
            f"Agent {self.name} - {self.id}: History compacted. "
            f"Summary: {'yes' if summary else 'no'}, preserved: {len(preserved)} messages."
        )

    def _compact_tool_outputs(self) -> bool:
        """Replace stale tool observation content with compact references.

        Iterates over messages outside the preserved tail and replaces large
        Observation messages with short placeholders.  The full content is
        stored in the agent's file store so the agent can retrieve it if needed.

        Returns:
            True if any compaction was performed.
        """
        preserve_n = self.summarization_config.preserve_last_messages
        threshold = self.summarization_config.compaction_token_threshold
        history_end = -preserve_n if preserve_n > 0 else len(self._prompt.messages)
        compactable = self._prompt.messages[self._history_offset : history_end]

        compacted_any = False
        for msg in compactable:
            if not (isinstance(msg, Message) and msg.role == MessageRole.USER and msg.content):
                continue
            if not msg.content.lstrip().startswith("Observation:"):
                continue

            msg_tokens = token_counter(
                model=self.llm.model,
                messages=[msg.model_dump(exclude={"metadata"}, exclude_none=True)],
            )
            if msg_tokens <= threshold:
                continue

            from dynamiq.utils import generate_uuid

            ref_id = generate_uuid()
            file_store = getattr(self, "file_store_backend", None)
            if file_store:
                file_store.store(f"compacted/{ref_id}.txt", msg.content.encode())
                msg.content = (
                    f"\nObservation: [Output compacted - ref:{ref_id}. "
                    f"Use file_read tool with path 'compacted/{ref_id}.txt' to retrieve full content.]\n"
                )
            else:
                msg.content = "\nObservation: [Output compacted to save context space.]\n"
            msg.static = True
            compacted_any = True

        if compacted_any:
            logger.info(f"Agent {self.name} - {self.id}: Compacted stale tool outputs.")
        return compacted_any

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
                text = message.content or ""
                if message.role == MessageRole.ASSISTANT:
                    history += f"-TOOL DESCRIPTION START-\n{text}\n-TOOL DESCRIPTION END-\n"
                elif message.role == MessageRole.USER:
                    history += f"-TOOL OUTPUT START-\n{text}\n-TOOL OUTPUT END-\n"

        return history
