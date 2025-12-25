from dynamiq.nodes.agents.base import AgentManager
from dynamiq.nodes.agents.prompts.orchestrators.base import PROMPT_TEMPLATE_BASE_HANDLE_INPUT
from dynamiq.nodes.agents.prompts.orchestrators.linear import (
    PROMPT_TEMPLATE_LINEAR_ASSIGN,
    PROMPT_TEMPLATE_LINEAR_FINAL,
    PROMPT_TEMPLATE_LINEAR_PLAN,
)


class LinearAgentManager(AgentManager):
    """
    A specialized AgentManager that manages tasks in a linear, sequential order.
    It uses predefined prompts to plan tasks, assign them to the appropriate agents,
    and compile the final result.
    """

    name: str = "Linear Manager"

    def __init__(self, **kwargs):
        """
        Initializes the LinearAgentManager and sets up the prompt blocks.
        """
        super().__init__(**kwargs)
        self._init_prompt_blocks()

    def _init_actions(self):
        """
        Extend default actions with 'respond'.
        """
        super()._init_actions()
        self._actions["handle_input"] = self._handle_input

    def _init_prompt_blocks(self):
        """
        Initializes the prompt blocks used in the task planning, assigning,
        and final answer generation processes.
        """
        super()._init_prompt_blocks()
        self.system_prompt_manager.update_blocks(
            {
                "plan": self._get_linear_plan_prompt(),
                "assign": self._get_linear_assign_prompt(),
                "final": self._get_linear_final_prompt(),
                "handle_input": self._get_linear_handle_input_prompt(),
            }
        )

    @staticmethod
    def _get_linear_plan_prompt() -> str:
        """
        Returns the prompt template for planning tasks.
        """
        return PROMPT_TEMPLATE_LINEAR_PLAN

    @staticmethod
    def _get_linear_assign_prompt() -> str:
        """
        Returns the prompt template for assigning tasks to agents.
        """
        return PROMPT_TEMPLATE_LINEAR_ASSIGN

    @staticmethod
    def _get_linear_final_prompt() -> str:
        """
        Returns the prompt template for generating the final answer.
        """
        return PROMPT_TEMPLATE_LINEAR_FINAL

    @staticmethod
    def _get_linear_handle_input_prompt() -> str:
        """Determines how to handle input, either by continuing the flow or providing a direct response."""
        return PROMPT_TEMPLATE_BASE_HANDLE_INPUT
