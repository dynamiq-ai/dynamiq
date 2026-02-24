from typing import Any, ClassVar, Literal

from litellm import token_counter
from pydantic import BaseModel, ConfigDict, Field

from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes import ErrorHandling, Node, NodeGroup
from dynamiq.nodes.agents.prompts.react.instructions import HISTORY_SUMMARIZATION_PROMPT_REPLACE
from dynamiq.nodes.node import NodeDependency, ensure_config
from dynamiq.prompts import Message, MessageRole
from dynamiq.prompts.prompts import Prompt, VisionMessage
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.utils.logger import logger

MERGE_SUMMARIES_PROMPT = (
    "The following are summaries of consecutive parts of a conversation. "
    "Merge them into a single cohesive summary, preserving all key decisions, "
    "important information, tool outputs, and unresolved tasks."
)

TOKEN_BUDGET_RATIO = 0.65


class ContextManagerInputSchema(BaseModel):
    """Input for ContextManagerTool.

    - notes: Verbatim content that must be preserved as-is and prepended to the summary.
    - messages: List of messages to summarize.
    """

    notes: str | None = Field(
        default=None,
        description=(
            "Optional notes to preserve verbatim (e.g., IDs, filenames, critical details). "
            "This will be prepended to the automatic summary."
        ),
    )

    messages: list[Message | VisionMessage] = Field(
        default=[],
        description="List of messages to summarize (conversation history).",
        json_schema_extra={"is_accessible_to_agent": False},
    )


class ContextManagerTool(Node):
    """
    A tool that generates a conversation summary.

    When called by the agent, this tool:
    1. Generates a summary of the conversation using its own LLM
    2. Returns the summary as tool result
    3. Agent then decides how to apply the summary.

    The tool doesn't modify the agent's state - it just generates and returns the summary.

    Attributes:
        group (Literal[NodeGroup.TOOLS]): The group this node belongs to.
        name (str): The name of the tool.
        description (str): Tool description with usage warning.
        error_handling (ErrorHandling): Configuration for error handling.
        llm (Node): LLM instance for generating summaries.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "Context Manager Tool"
    description: str = (
        "Generates a conversation summary to help manage context.\n\n"
        "WARNING: This tool will trigger context compression. Before calling it,\n"
        "save any necessary information because previous messages will be removed.\n"
    )

    error_handling: ErrorHandling = Field(default_factory=lambda: ErrorHandling(timeout_seconds=60))
    llm: Node = Field(..., description="LLM instance for generating summaries")

    model_config = ConfigDict(arbitrary_types_allowed=True)
    input_schema: ClassVar[type[ContextManagerInputSchema]] = ContextManagerInputSchema

    def init_components(self, connection_manager: ConnectionManager | None = None) -> None:
        """Initialize components for the tool."""
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        # Initialize the LLM if it is a postponed component
        if self.llm.is_postponed_component_init:
            self.llm.init_components(connection_manager)

    def reset_run_state(self):
        """Reset the intermediate steps (run_depends) of the node."""
        self._run_depends = []

    @property
    def to_dict_exclude_params(self) -> dict:
        """
        Property to define which parameters should be excluded when converting the class instance to a dictionary.

        Returns:
            dict: A dictionary defining the parameters to exclude.
        """
        return super().to_dict_exclude_params | {"llm": True}

    def to_dict(self, **kwargs) -> dict:
        """Converts the instance to a dictionary.

        Returns:
            dict: A dictionary representation of the instance.
        """
        data = super().to_dict(**kwargs)
        data["llm"] = self.llm.to_dict(**kwargs)
        return data

    def _get_token_budget(self) -> int:
        """Return the max input tokens available for summarization content.

        Reserves space for the summarization prompt itself and the LLM output.
        """
        return int(self.llm.get_token_limit() * TOKEN_BUDGET_RATIO)

    def _count_message_tokens(self, messages: list[Message | VisionMessage]) -> int:
        """Count tokens for a list of messages using the summarization LLM's tokenizer."""
        return token_counter(
            model=self.llm.model,
            messages=[m.model_dump(exclude={"metadata"}) for m in messages],
        )

    def _truncate_message(self, msg: Message | VisionMessage, budget: int) -> Message | VisionMessage:
        """Truncate a single message so it fits within *budget* tokens.

        Keeps a prefix of the content and appends a truncation marker.
        VisionMessages are returned as-is (image content can't be meaningfully truncated).
        """
        if isinstance(msg, VisionMessage):
            return msg

        msg_tokens = self._count_message_tokens([msg])
        if msg_tokens <= budget:
            return msg

        content = msg.content
        ratio = budget / msg_tokens
        # Conservative cut — leave room for the truncation marker
        cut_point = max(1, int(len(content) * ratio * 0.9))
        truncated_content = content[:cut_point] + "\n\n[... truncated due to context limit ...]"
        logger.warning(
            f"Context Manager Tool: Truncating oversized message "
            f"({msg_tokens} tokens > {budget} budget). Kept ~{cut_point}/{len(content)} chars."
        )
        return msg.model_copy(update={"content": truncated_content})

    def _split_messages_into_chunks(
        self,
        messages: list[Message | VisionMessage],
        budget: int,
    ) -> list[list[Message | VisionMessage]]:
        """Split messages into chunks that each fit within *budget* tokens.

        Messages are kept in order. A single message that exceeds the budget
        is truncated to fit.
        """
        chunks: list[list[Message | VisionMessage]] = []
        current_chunk: list[Message | VisionMessage] = []
        current_tokens = 0

        for msg in messages:
            msg_tokens = self._count_message_tokens([msg])

            if msg_tokens > budget:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = []
                    current_tokens = 0
                truncated = self._truncate_message(msg, budget)
                chunks.append([truncated])
                continue

            if current_chunk and current_tokens + msg_tokens > budget:
                chunks.append(current_chunk)
                current_chunk = [msg]
                current_tokens = msg_tokens
            else:
                current_chunk.append(msg)
                current_tokens += msg_tokens

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _call_llm_for_summary(
        self,
        messages: list[Message | VisionMessage],
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> str:
        """Send *messages* to the LLM and return the generated text."""
        llm_result = self.llm.run(
            input_data={},
            prompt=Prompt(messages=messages),
            config=config,
            **(kwargs | {"parent_run_id": kwargs.get("run_id"), "run_depends": []}),
        )
        self._run_depends = [NodeDependency(node=self.llm).to_dict(for_tracing=True)]

        if llm_result.status != RunnableStatus.SUCCESS:
            error_msg = llm_result.error.message if llm_result.error else "Unknown error"
            raise ValueError(f"Context Manager Tool: LLM failed to generate summary: {error_msg}")

        summary = llm_result.output.get("content", "")
        if not summary:
            raise ValueError("Context Manager Tool: LLM returned empty summary.")
        return summary

    def _summarize_replace_history(
        self,
        messages: list[Message | VisionMessage],
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> str:
        """Generate a complete summary of the conversation (replace mode).

        When the conversation history exceeds the LLM's context window, the
        messages are split into chunks that fit, each chunk is summarised
        independently, and the per-chunk summaries are merged into one final
        summary.  If the merged text is still too large, a second merge pass
        is applied.

        Args:
            messages: List of messages to summarize.
            config: Configuration for the run.
            **kwargs: Additional parameters.

        Returns:
            str: The generated summary.
        """
        budget = self._get_token_budget()
        message_tokens = self._count_message_tokens(messages)

        if message_tokens <= budget:
            logger.info("Context Manager Tool: History fits in context, single-pass summary.")
            summary_messages = messages + [
                Message(content=HISTORY_SUMMARIZATION_PROMPT_REPLACE, role=MessageRole.USER, static=True),
            ]
            summary = self._call_llm_for_summary(summary_messages, config, **kwargs)
            logger.info(f"Context Manager Tool: Summary generated. Length: {len(summary)}")
            return summary

        chunks = self._split_messages_into_chunks(messages, budget)
        logger.info(
            f"Context Manager Tool: History ({message_tokens} tokens) exceeds budget "
            f"({budget} tokens). Splitting into {len(chunks)} chunks."
        )

        chunk_summaries: list[str] = []
        for idx, chunk in enumerate(chunks):
            logger.info(f"Context Manager Tool: Summarizing chunk {idx + 1}/{len(chunks)}.")
            chunk_messages = chunk + [
                Message(content=HISTORY_SUMMARIZATION_PROMPT_REPLACE, role=MessageRole.USER, static=True),
            ]
            chunk_summaries.append(self._call_llm_for_summary(chunk_messages, config, **kwargs))

        combined = "\n\n---\n\n".join(chunk_summaries)
        combined_messages = [
            Message(content=combined, role=MessageRole.USER, static=True),
            Message(content=MERGE_SUMMARIES_PROMPT, role=MessageRole.USER, static=True),
        ]

        combined_tokens = self._count_message_tokens(combined_messages)
        if combined_tokens > budget:
            logger.info(
                "Context Manager Tool: Merged summaries still exceed budget "
                f"({combined_tokens} > {budget}). Running additional merge pass."
            )
            merge_chunks = self._split_messages_into_chunks(
                [Message(content=s, role=MessageRole.USER, static=True) for s in chunk_summaries],
                budget,
            )
            re_summaries: list[str] = []
            for merge_chunk in merge_chunks:
                merge_chunk.append(Message(content=MERGE_SUMMARIES_PROMPT, role=MessageRole.USER, static=True))
                re_summaries.append(self._call_llm_for_summary(merge_chunk, config, **kwargs))
            combined = "\n\n".join(re_summaries)
            combined_messages = [
                Message(content=combined, role=MessageRole.USER, static=True),
                Message(content=MERGE_SUMMARIES_PROMPT, role=MessageRole.USER, static=True),
            ]

        summary = self._call_llm_for_summary(combined_messages, config, **kwargs)
        logger.info(f"Context Manager Tool: Chunked summary generated. Length: {len(summary)}")
        return summary

    def execute(
        self, input_data: ContextManagerInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Generate conversation summary from provided messages.

        Returns:
            dict[str, Any]:
                - content: The generated summary
                - notes: Optional notes to preserve
        """
        config = ensure_config(config)
        self.reset_run_state()
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        if not self.llm:
            raise ValueError("Context Manager Tool: No LLM configured.")

        if not input_data.messages:
            raise ValueError("Context Manager Tool: No messages provided to summarize.")

        logger.info(f"Context Manager Tool: Generating summary for {len(input_data.messages)} messages.")

        summary_result = self._summarize_replace_history(
            input_data.messages,
            config,
            **kwargs,
        )

        # Return summary with optional notes
        result_content = summary_result
        if input_data.notes:
            result_content = f"Notes: {input_data.notes}\n\n{summary_result}"

        return {
            "content": result_content,
        }
