from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes import ErrorHandling, Node, NodeGroup
from dynamiq.nodes.agents.exceptions import ParsingError
from dynamiq.nodes.agents.prompts.react.instructions import HISTORY_SUMMARIZATION_PROMPT_REPLACE
from dynamiq.nodes.agents.utils import SummarizationConfig, XMLParser
from dynamiq.nodes.node import ensure_config
from dynamiq.prompts import Message, MessageRole
from dynamiq.prompts.prompts import Prompt, VisionMessage
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.utils.logger import logger


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

    messages: list[Message] = Field(
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
        max_summarization_retries (int): Maximum retry attempts for summary extraction.
        summarization_config (Any): Summarization configuration from agent (includes system_prompt).
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
    max_summarization_retries: int = Field(default=3, ge=1, description="Maximum retry attempts for summary extraction")
    summarization_config: SummarizationConfig = Field(..., description="Summarization configuration from agent")

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
    def to_dict_exclude_params(self):
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

    def _summarize_replace_history(
        self,
        messages: list[Message | VisionMessage],
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> str:
        """
        Generate a complete summary of the conversation (replace mode).

        Args:
            messages: List of messages to summarize
            config: Configuration for the run
            **kwargs: Additional parameters

        Returns:
            str: The generated summary
        """
        logger.info("Context Manager Tool: Generating summary (replace mode).")

        # Build summary request messages with constant prompt
        summary_messages = messages + [
            Message(
                content=HISTORY_SUMMARIZATION_PROMPT_REPLACE,
                role=MessageRole.USER,
                static=True,
            ),
        ]

        # Attempt to generate and extract summary
        output = ""  # Default in case loop doesn't execute
        for attempt in range(self.max_summarization_retries):
            llm_result = self.llm.run(
                input_data={},
                prompt=Prompt(messages=summary_messages),
                config=config,
            )

            if llm_result.status != RunnableStatus.SUCCESS:
                error_msg = llm_result.error.message if llm_result.error else "Unknown error"
                logger.warning(f"Context Manager Tool: LLM call failed on attempt {attempt + 1}: {error_msg}")
                continue

            output = llm_result.output.get("content", "")

            try:
                # Try to extract summary from XML tags
                summary = XMLParser.extract_first_tag_regex(output, ["summary"])

                if summary:
                    logger.info(f"Context Manager Tool: Summary generated successfully. Length: {len(summary)}")
                    return summary
                else:
                    # Try alternative parsing
                    parsed_data = XMLParser.parse(
                        f"<root>{output}</root>",
                        required_tags=["summary"],
                        optional_tags=[],
                    )
                    summary = parsed_data.get("summary", "")
                    if summary:
                        logger.info(f"Context Manager Tool: Summary generated successfully. Length: {len(summary)}")
                        return summary

            except ParsingError as e:
                logger.warning(f"Context Manager Tool: Failed to extract summary on attempt {attempt + 1}: {e}")
                # Add feedback for next attempt
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

        # If all attempts failed, use raw output
        logger.warning(
            f"Context Manager Tool: Could not extract summary after {self.max_summarization_retries} attempts. "
            "Using raw output."
        )
        return output

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
            error_msg = "Context Manager Tool: No LLM configured."
            logger.error(error_msg)
            return {"content": error_msg}

        if not input_data.messages:
            error_msg = "Context Manager Tool: No messages provided to summarize."
            logger.error(error_msg)
            return {"content": error_msg}

        logger.info(
            f"Context Manager Tool: Generating summary for {len(input_data.messages)} messages "
        )

        try:
            # Generate summary
            summary_result = self._summarize_replace_history(
                input_data.messages,
                config,
            )

            # Return summary with optional notes
            result_content = summary_result
            if input_data.notes:
                result_content = f"Notes: {input_data.notes}\n\n{summary_result}"

            return {
                "content": result_content,
            }

        except Exception as e:
            error_msg = f"Context Manager Tool: Error generating summary: {e}"
            logger.error(error_msg)
            return {"content": error_msg}
