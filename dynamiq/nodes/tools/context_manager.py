from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes import ErrorHandling, Node, NodeGroup
from dynamiq.nodes.llms import BaseLLM
from dynamiq.nodes.node import NodeDependency, ensure_config
from dynamiq.prompts import Message, Prompt
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.utils.logger import logger

CONTEXT_MANAGER_PROMPT_TEMPLATE = """
You are a context compression assistant for an AI agent.

IMPORTANT: The agent will delete previous message history after this step. You MUST preserve all
essential information needed to continue the task successfully.

Task:
- Produce a detailed summary that replaces the prior message history.
- Keep only what is necessary to proceed: reasoning overview, current subtasks, saved information and files, next steps,
 additional notes.
- Omit chit-chat and non-essential details. Use clear, structured formatting.

History to compress:
{history}

Output strictly in this structure:

## Reasoning overview of what is reasoning flow

## Current Subtasks
- [ordered bullets: subtask -> status]

## Saved information and files
- Inform about filesystem state and files that are saved (if available)

## Next Steps
- [ordered bullets: next step -> status]

## Additional Notes:
Any other information that is important to keep in mind and not lost.

"""


class ContextManagerInputSchema(BaseModel):
    """Input for ContextManagerTool.

    - history: The recent conversation/messages to compress. Can be a single string or list of strings.
    - is_history_preserved: Preserve the history with summarization. If False, the history will not be preserved,
     only notes will.
    - notes: Verbatim content that must be preserved as-is (not processed by LLM) and prepended to the result.
    """

    history: list[Message] | None = Field(
        ..., description="Conversation history to be summarized and used to replace prior messages"
    )

    is_history_preserved: bool = Field(
        default=True,
        description="Preserve the history with summarization. If False, the history will not be preserved,"
        " only notes will.",
    )

    notes: str | None = Field(
        default=None,
        description=(
            "Verbatim content to preserve as-is (e.g., IDs, filenames, critical details). "
            "This will be prepended unchanged to the output and NOT sent to the LLM."
        ),
    )


class ContextManagerTool(Node):
    """
    A tool to prune previous message history and replace it with a concise summary.

    IMPORTANT: Before calling this tool, ensure any necessary details are explicitly saved
    (e.g., files, pinned notes, or artifacts). This tool is intended to remove previous messages
    and keep only a structured summary to tighten context and focus on the active subtask.

    Attributes:
        group (Literal[NodeGroup.TOOLS]): The group this node belongs to.
        name (str): The name of the tool.
        description (str): Tool description with usage warning.
        llm (BaseLLM): The LLM used to produce the compressed summary.
        error_handling (ErrorHandling): Configuration for error handling.
        prompt_template (str): Prompt template guiding the summarization.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "Context Manager Tool"
    description: str = (
        "Cleans prior message history and replaces it with a concise, self-contained summary.\n\n"
        "WARNING: Before calling this tool, the agent must save any necessary information (f.e in FileStore),\n"
        "because previous messages will be removed and replaced by the summary. "
        "You can also provide notes to the tool to preserve important information without being processed by the LLM. "
        "Make sure to provide all necessary information for the agent to stay on track and"
        " not lose any important details. "
        "You can also disable history preservation, only notes will be preserved. "
        "Disable history when you don't care about the history and only want to preserve notes."
    )

    llm: BaseLLM = Field(..., description="LLM used to produce the compressed context summary")
    error_handling: ErrorHandling = Field(default_factory=lambda: ErrorHandling(timeout_seconds=600))
    prompt_template: str = Field(
        default=CONTEXT_MANAGER_PROMPT_TEMPLATE, description="Prompt template for context compression"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)
    input_schema: ClassVar[type[ContextManagerInputSchema]] = ContextManagerInputSchema

    def init_components(self, connection_manager: ConnectionManager | None = None) -> None:
        """Initialize components for the tool."""
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        if self.llm.is_postponed_component_init:
            self.llm.init_components(connection_manager)

    def reset_run_state(self):
        """Reset the intermediate steps (run_depends) of the node."""
        self._run_depends = []

    @property
    def to_dict_exclude_params(self) -> dict:
        """Exclude LLM object during serialization."""
        return super().to_dict_exclude_params | {"llm": True}

    def to_dict(self, **kwargs) -> dict:
        data = super().to_dict(**kwargs)
        data["llm"] = self.llm.to_dict(**kwargs)
        return data

    def _build_prompt(self, history: list[Message]) -> str:
        formatted_history = "\n\n---\n\n".join([f"{m.role}: {str(m.content)}" for m in history])
        return self.prompt_template.format(history=formatted_history)

    def _summarize_history(self, history: list[Message], config: RunnableConfig, **kwargs) -> str:
        prompt_content = self._build_prompt(history)

        result = self.llm.run(
            input_data={},
            prompt=Prompt(messages=[Message(role="user", content=prompt_content, static=True)]),
            config=config,
            **(kwargs | {"parent_run_id": kwargs.get("run_id"), "run_depends": []}),
        )

        self._run_depends = [NodeDependency(node=self.llm).to_dict(for_tracing=True)]

        if result.status != RunnableStatus.SUCCESS:
            raise ValueError("LLM execution failed during context compression")

        return result.output.get("content", "").strip()

    def execute(
        self, input_data: ContextManagerInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Summarize the provided history and emit an instruction to replace prior messages with the summary.

        Returns:
            dict[str, Any]:
                - content: human-readable status message
                - summary: the compressed summary text
                - keep_last_n: advisory hint for UI/agent to keep last N messages
                - replacement_message: suggested system message to insert as new context root
                - instructions_for_agent: explicit instructions for applying the change
        """
        config = ensure_config(config)
        self.reset_run_state()
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        summary = ""

        if input_data.is_history_preserved:
            summary = self._summarize_history(input_data.history, config, **kwargs)
            summary = f"\nContext compressed; Summary:\n {summary}"

        if input_data.notes:
            summary = f"Notes: {input_data.notes}\n\n{summary}"

        logger.debug(f"Tool {self.name} - {self.id}: context compression completed, summary length: {len(summary)}")

        return {"content": summary}


def _apply_context_manager_tool_effect(prompt: Prompt, tool_result: Any, history_offset: int) -> None:
    """Apply context cleaning effect after ContextManagerTool call.

    Keeps default prefix (up to history_offset), replaces the rest with a copy of the last prefix message,
    and appends an observation with the tool_result summary.
    """

    try:
        new_messages = prompt.messages[:history_offset]
        if new_messages:
            new_messages.append(prompt.messages[-1].copy())
        prompt.messages = new_messages

    except Exception as e:
        logger.error(f"Error applying context manager tool effect: {e}")
