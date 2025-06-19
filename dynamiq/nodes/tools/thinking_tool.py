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
    description: str = """## Structured Reasoning Tool
### Purpose
Process thoughts and reasoning in a structured way to improve decision-making and problem-solving clarity.

### When to Use
- Before making important decisions or taking actions
- When analyzing complex problems or multi-step processes
- To organize thoughts and validate reasoning logic
- When reviewing tool results for accuracy and completeness
- For planning next steps in complex workflows
- To identify assumptions and potential gaps in thinking

### Key Capabilities
- Structured thought analysis and organization
- Break down complex problems into manageable components
- Identify key insights, assumptions, and knowledge gaps
- Plan sequential reasoning and action steps
- Validate logic and decision pathways
- Maintain thinking context across sessions (when memory enabled)

### Required Parameters
- **thought** (string): The idea, reasoning, or problem to analyze

### Optional Parameters
- **context** (string): Additional background information or constraints
- **focus** (string): Specific analysis area (planning, analysis, problem-solving, etc.)

### Usage Examples
#### Problem Analysis
```json
{
  "thought": "I need to decide between two database solutions for handling user data",
  "context": "We expect 100k users initially, growing to 1M+ users",
  "focus": "decision-making"
}
```

#### Action Planning
```json
{
  "thought": "The API integration failed with a 401 error, need to troubleshoot",
  "context": "Using OAuth2 authentication, worked yesterday",
  "focus": "problem-solving"
}
```

#### Requirement Validation
```json
{
  "thought": "User wants to export data but hasn't specified format or scope",
  "context": "They mentioned 'all customer data' but that could mean different things",
  "focus": "requirement-gathering"
}
```

### Analysis Framework
The tool provides structured output including:
- **Analysis**: Clarification and organization of the main thought
- **Key Components**: Breakdown of complex aspects
- **Insights & Observations**: Important patterns, assumptions, or gaps
- **Next Steps**: Concrete actions or further considerations
- **Summary**: Clear, actionable conclusion

### Best Practices
1. **Use before major decisions** to validate reasoning
2. **Break down complex thoughts** into smaller components
3. **Include relevant context** for better analysis
4. **Specify focus areas** for targeted analysis
5. **Review assumptions** and identify potential biases
6. **Use iteratively** for multi-step problem solving
7. **Document insights** for future reference
"""

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
