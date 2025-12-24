from dynamiq.nodes.agents.base import AgentManager
from dynamiq.nodes.agents.prompts.orchestrators.graph import (
    PROMPT_TEMPLATE_GRAPH_ASSIGN,
    PROMPT_TEMPLATE_GRAPH_HANDLE_INPUT,
    PROMPT_TEMPLATE_GRAPH_PLAN,
)


class GraphAgentManager(AgentManager):
    """A graph agent manager that coordinates graph flow execution."""

    name: str = "Graph Manager"

    def __init__(self, **kwargs):
        """Initialize the GraphAgentManager and set up prompt templates."""
        super().__init__(**kwargs)
        self._init_prompt_blocks()

    def _init_prompt_blocks(self):
        """Initialize the prompt blocks with finding next state, actions and final answer prompts."""
        super()._init_prompt_blocks()
        self.system_prompt_manager.update_blocks(
            {
                "plan": self._get_next_state_prompt(),
                "assign": self._get_actions_prompt(),
                "handle_input": self._get_graph_handle_input_prompt(),
            }
        )

    @staticmethod
    def _get_graph_handle_input_prompt() -> str:
        """Determines how to handle input, either by continuing the flow or providing a direct response."""
        return PROMPT_TEMPLATE_GRAPH_HANDLE_INPUT

    @staticmethod
    def _get_next_state_prompt() -> str:
        """Return next step prompt template."""
        return PROMPT_TEMPLATE_GRAPH_PLAN

    @staticmethod
    def _get_actions_prompt() -> str:
        """Return actions prompt template."""
        return PROMPT_TEMPLATE_GRAPH_ASSIGN
