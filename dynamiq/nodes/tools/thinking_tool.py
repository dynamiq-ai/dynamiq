from copy import deepcopy
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes import ErrorHandling, Node, NodeGroup
from dynamiq.nodes.llms import BaseLLM
from dynamiq.nodes.node import NodeDependency, ensure_config
from dynamiq.prompts import Message, Prompt
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.utils.logger import logger

THINKING_PROMPT_TEMPLATE = """You are a thinking assistant that helps
process thoughts and reasoning in a structured way.

Your role is to act as a cognitive scratchpad,
helping to:
- Clarify and organize complex thoughts
- Break down problems into manageable components
- Identify key insights, assumptions, and gaps
- Structure reasoning processes
- Plan next steps and actions

Current thought to process:
<thought>
{thought}
</thought>

{context_section}

Please provide a structured analysis following this format:

## Analysis
[Clarify and organize the main thought]

## Key Components
[Break down complex aspects if applicable]

## Insights & Observations
[Identify important patterns, assumptions, or gaps]

## Next Steps
[Suggest concrete actions or further considerations]

## Summary
[Provide a clear, actionable conclusion]

Focus on being thorough, logical, and helpful in your analysis."""  # noqa E501


class ThinkingInputSchema(BaseModel):
    thought: str = Field(..., description="The thought, idea, or reasoning to process and analyze")
    context: str = Field(default="", description="Additional context or background information")
    focus: str = Field(default="general", description="Specific focus area (planning, analysis, problem-solving, etc.)")


class ThinkingTool(Node):
    """
    A tool for structured thinking and reasoning processes.

    This tool helps agents process thoughts in a structured way, providing a cognitive
    scratchpad for complex reasoning, planning, and analysis. The agent's LLM will be
    used automatically when this tool is called by an agent.

    Attributes:
        group (Literal[NodeGroup.TOOLS]): The group this node belongs to.
        name (str): The name of the tool.
        description (str): A description of the tool's functionality.
        llm (BaseLLM): The LLM to use for processing thoughts.
        error_handling (ErrorHandling): Configuration for error handling.
        prompt_template (str): The prompt template used for thinking processes.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "Thinking Tool"
    description: str = """Analyzes thoughts and reasoning processes to improve decision-making clarity and identify gaps in logic.

Key Capabilities:
- Structured analysis of complex thoughts and problems
- Component breakdown with insights and next steps
- Context-aware analysis with optional memory support
- Sequential reasoning validation and action planning

Usage Strategy:
- Use before making important decisions or actions
- Analyze complex multi-step problems systematically
- Validate reasoning logic and identify assumptions
- Plan next steps in complex workflows with clarity

Parameter Guide:
- thought: The idea, reasoning, or problem to analyze (required)
- context: Background information or constraints
- focus: Analysis area (planning, problem-solving, decision-making)
- memory_enabled: Maintain history of previous thoughts

Examples:
- {"thought": "Should we implement feature X?", "focus": "decision-making"}
- {"thought": "API integration failed with 401 error", "context": "OAuth2 auth"}
- {"thought": "Choose database solutions", "context": "100k users, scaling"}"""  # noqa E501

    llm: BaseLLM = Field(..., description="LLM to use for thinking processes")

    error_handling: ErrorHandling = Field(default_factory=lambda: ErrorHandling(timeout_seconds=600))

    prompt_template: str = Field(
        default=THINKING_PROMPT_TEMPLATE, description="The prompt template for the thinking process"
    )

    memory_enabled: bool = Field(
        default=False, description="Whether to maintain memory of previous thoughts in this session"
    )
    max_thoughts_in_memory: int = Field(
        default=3,
        description="Number of recent thoughts to keep in memory when memory is enabled",
    )
    max_thought_chars: int = Field(
        default=300,
        description="Maximum characters of each thought to display in memory when memory is enabled",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)
    input_schema: ClassVar[type[ThinkingInputSchema]] = ThinkingInputSchema

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._thought_history: list[dict] = []

    def init_components(self, connection_manager: ConnectionManager | None = None) -> None:
        """Initialize the components of the tool."""
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)

        if self.llm.is_postponed_component_init:
            self.llm.init_components(connection_manager)

    def reset_run_state(self):
        """Reset the intermediate steps (run_depends) of the node."""
        self._run_depends = []

    @property
    def to_dict_exclude_params(self) -> dict:
        """Property to define which parameters should be excluded when converting to dictionary."""
        return super().to_dict_exclude_params | {"llm": True}

    def to_dict(self, **kwargs) -> dict:
        """Convert the tool to a dictionary representation."""
        data = super().to_dict(**kwargs)
        data["llm"] = self.llm.to_dict(**kwargs)
        return data

    def _build_context_section(self, context: str, focus: str) -> str:
        """Build the context section for the prompt."""
        sections = []

        if context:
            sections.append(f"Additional context:\n{context}")

        if focus and focus != "general":
            sections.append(f"Focus area: {focus}")

        if self.memory_enabled and self._thought_history:
            recent_thoughts = self._thought_history[-self.max_thoughts_in_memory :]
            history_text = "\n".join(
                [
                    f"- {i + 1}. {thought['thought'][:self.max_thought_chars]}{'...' if len(thought['thought']) > self.max_thought_chars else ''}"  # noqa E501
                    for i, thought in enumerate(recent_thoughts)
                ]
            )
            sections.append(f"Recent thinking history:\n{history_text}")

        return "\n\n".join(sections) if sections else ""

    def execute(
        self, input_data: ThinkingInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Execute the thinking tool on the input data.

        This method processes a thought through structured analysis, helping to clarify,
        organize, and develop the reasoning around the given input.

        Args:
            input_data (ThinkingInputSchema): Input containing thought, context, and focus
            config (RunnableConfig, optional): The configuration for running the tool
            **kwargs: Additional keyword arguments

        Returns:
            dict[str, Any]: A dictionary containing the analysis, original thought, and metadata
        """
        config = ensure_config(config)
        self.reset_run_state()
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        thought = input_data.thought
        context = input_data.context
        focus = input_data.focus

        logger.debug(f"Tool {self.name} - {self.id}: started thinking process for thought: '{thought[:100]}...'")

        context_section = self._build_context_section(context, focus)

        prompt_content = self.prompt_template.format(thought=thought, context_section=context_section)

        logger.debug(f"Tool {self.name} - {self.id}: prompt content:\n{prompt_content}")

        result = self.llm.run(
            input_data={},
            prompt=Prompt(messages=[Message(role="user", content=prompt_content, static=True)]),
            config=config,
            **(kwargs | {"parent_run_id": kwargs.get("run_id")}),
        )

        logger.debug(f"Tool {self.name} - {self.id}: result status: {result.output}")

        self._run_depends = [NodeDependency(node=self.llm).to_dict()]

        if result.status != RunnableStatus.SUCCESS:
            raise ValueError("LLM execution failed during thinking process")

        analysis = result.output["content"]

        if self.memory_enabled:
            self._thought_history.append(
                {
                    "thought": thought,
                    "context": context,
                    "focus": focus,
                    "analysis": analysis,
                    "timestamp": kwargs.get("run_id", "unknown"),
                }
            )

        logger.debug(
            f"Tool {self.name} - {self.id}: completed thinking process, " f"analysis length: {len(analysis)} characters"
        )

        return {
            "content": analysis,
            "original_thought": thought,
            "context_used": context,
            "focus_area": focus,
            "thinking_session_count": len(self._thought_history) if self.memory_enabled else None,
        }

    def clear_memory(self) -> None:
        """Clear the thinking history memory."""
        self._thought_history.clear()
        logger.debug(f"Tool {self.name} - {self.id}: cleared thinking history memory")

    def get_thought_history(self) -> list[dict]:
        """Get the current thought history."""
        return deepcopy(self._thought_history) if self.memory_enabled else []
