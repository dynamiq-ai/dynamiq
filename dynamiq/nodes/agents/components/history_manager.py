"""History and context management mixin for agents."""

from litellm import token_counter

from dynamiq.nodes.agents.prompts.secondary_instructions import AGENT_NOTES_FILENAME, NOTES_REVISIT_INSTRUCTION_TEMPLATE
from dynamiq.nodes.agents.utils import extract_message_text
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
    - sandbox_backend (optional): sandbox used to resolve the persistent
      notes file path.
    """

    def get_notes_file_path(self) -> str | None:
        """Path of the dedicated persistent-notes file, or None without a sandbox.

        The file lives in the sandbox so exact values survive context
        compaction; the path is advertised in the system prompt and
        re-surfaced in the compaction observation.
        """
        sandbox = getattr(self, "sandbox_backend", None)
        if sandbox is not None:
            return f"{sandbox.base_path.rstrip('/')}/{AGENT_NOTES_FILENAME}"
        return None

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
        force: bool = False,
    ) -> tuple[list[Message | VisionMessage], list[Message | VisionMessage]]:
        """Split conversation history into messages to summarize and messages to preserve.

        Walks from newest to oldest, greedily keeping messages that fit within
        ``max_preserved_tokens``.  Everything else goes to the summarize bucket.

        Args:
            force: Summarize even when the local token estimate says the history fits.
                Set when a provider has already rejected the prompt as too long, which
                means that estimate was wrong and preserving the whole tail would make
                compaction a no-op exactly when it is needed.

        Returns:
            Tuple of (to_summarize, to_preserve).
        """
        conversation_history = self._prompt.messages[self._history_offset :]

        n_tokens, preserve_count = 0, 0
        for msg in reversed(conversation_history):
            msg_tokens = token_counter(
                model=self.llm.model,
                messages=[msg.model_dump(exclude={"metadata"})],
            )
            if n_tokens + msg_tokens > self.summarization_config.max_preserved_tokens:
                break
            n_tokens += msg_tokens
            preserve_count += 1

        if preserve_count > 0:
            to_summarize = conversation_history[:-preserve_count]
            to_preserve = [m.copy() for m in conversation_history[-preserve_count:]]
        else:
            to_summarize = conversation_history
            to_preserve = []

        # Drop orphan tool replies whose assistant ended up in to_summarize —
        # an unpaired role:tool message at the boundary would 400 the next request.
        while to_preserve and to_preserve[0].role == MessageRole.TOOL:
            to_summarize = list(to_summarize) + [to_preserve.pop(0)]

        if to_summarize == [] and (force or self.is_token_limit_exceeded()):
            to_summarize = conversation_history
            to_preserve = []

        return to_summarize, to_preserve

    def _compact_history(
        self,
        summary: str | None = None,
        preserved: list[Message | VisionMessage] | None = None,
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
            preserved: Pre-computed preserved messages from ``_split_history``.
                When supplied, skips the redundant ``_split_history`` call.
        """
        if preserved is None:
            _, preserved = self._split_history()

        self._prompt.messages = self._prompt.messages[: self._history_offset]

        if summary:
            if pinned_content is not None:
                preserved_has_pinned = any(extract_message_text(m) == pinned_content for m in preserved)
                if not preserved_has_pinned:
                    summary = f"{summary}\n\nOriginal request: {pinned_content}"

            if notes_path := self.get_notes_file_path():
                summary = f"{summary}\n\n{NOTES_REVISIT_INSTRUCTION_TEMPLATE.format(notes_path=notes_path)}"

            self._prompt.messages.append(
                Message(role=MessageRole.USER, content=f"Observation: {summary}\n", static=True)
            )

        self._prompt.messages.extend(preserved)
        logger.info(
            f"Agent {self.name} - {self.id}: History compacted. "
            f"Summary: {'yes' if summary else 'no'}, preserved: {len(preserved)} messages."
        )

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
                    history += f"-TOOL DESCRIPTION START-\n{message.content}\n"
                    for tool_call in message.tool_calls or []:
                        fn = tool_call.get("function", {})
                        history += f"Tool: {fn.get('name', '')}\nArguments: {fn.get('arguments', '')}\n"
                    history += "-TOOL DESCRIPTION END-\n"
                elif message.role in (MessageRole.USER, MessageRole.TOOL):
                    history += f"-TOOL OUTPUT START-\n{message.content}\n-TOOL OUTPUT END-\n"

        return history
