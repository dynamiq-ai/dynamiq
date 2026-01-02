from dynamiq.nodes.agents.base import AgentManager
from dynamiq.nodes.agents.prompts.orchestrators.adaptive import (
    PROMPT_TEMPLATE_ADAPTIVE_FINAL,
    PROMPT_TEMPLATE_ADAPTIVE_PLAN,
    PROMPT_TEMPLATE_ADAPTIVE_REFLECT,
    PROMPT_TEMPLATE_ADAPTIVE_RESPOND,
)
from dynamiq.nodes.agents.prompts.orchestrators.base import PROMPT_TEMPLATE_BASE_HANDLE_INPUT
from dynamiq.prompts import Message, MessageRole
from dynamiq.runnables import RunnableConfig
from dynamiq.types.streaming import StreamingMode


class AdaptiveAgentManager(AgentManager):
    """An adaptive agent manager that coordinates specialized agents to complete complex tasks."""

    name: str = "Adaptive Manager"

    def __init__(self, **kwargs):
        """Initialize the AdaptiveAgentManager and set up prompt templates."""
        super().__init__(**kwargs)
        self._init_prompt_blocks()

    def _init_actions(self):
        """Extend the default actions with 'respond'."""
        super()._init_actions()  #
        self._actions["respond"] = self._respond
        self._actions["reflect"] = self._reflect

    def _init_prompt_blocks(self):
        """Initialize the prompt blocks with adaptive plan and final prompts."""
        super()._init_prompt_blocks()
        self.system_prompt_manager.update_blocks(
            {
                "plan": self._get_adaptive_plan_prompt(),
                "final": self._get_adaptive_final_prompt(),
                "respond": self._get_adaptive_respond_prompt(),
                "reflect": self._get_adaptive_reflect_prompt(),
                "handle_input": self._get_adaptive_handle_input_prompt(),
            }
        )

    @staticmethod
    def _get_adaptive_plan_prompt() -> str:
        """Return the adaptive plan prompt template."""
        return PROMPT_TEMPLATE_ADAPTIVE_PLAN

    @staticmethod
    def _get_adaptive_handle_input_prompt() -> str:
        """Determines how to handle input, either by continuing the flow or providing a direct response."""
        return PROMPT_TEMPLATE_BASE_HANDLE_INPUT

    @staticmethod
    def _get_adaptive_final_prompt() -> str:
        """Return the adaptive final answer prompt template."""
        return PROMPT_TEMPLATE_ADAPTIVE_FINAL

    @staticmethod
    def _get_adaptive_respond_prompt() -> str:
        """Return the adaptive clarify prompt template."""
        return PROMPT_TEMPLATE_ADAPTIVE_RESPOND

    @staticmethod
    def _get_adaptive_reflect_prompt() -> str:
        """Return the adaptive reflect prompt template."""
        return PROMPT_TEMPLATE_ADAPTIVE_REFLECT

    def _reflect(self, config: RunnableConfig, **kwargs) -> str:
        """Executes the 'reflect' action."""
        prompt = self.system_prompt_manager.render_block(
            "reflect", **(self.system_prompt_manager._prompt_variables | kwargs)
        )
        llm_result = self._run_llm([Message(role=MessageRole.USER, content=prompt)], config, **kwargs).output["content"]
        if self.streaming.enabled and self.streaming.mode == StreamingMode.ALL:
            return self.stream_content(
                content=llm_result,
                step="manager_reflection",
                source=self.name,
                config=config,
                **kwargs
            )
        return llm_result

    def _respond(self, config: RunnableConfig, **kwargs) -> str:
        """Executes the 'respond' action."""
        prompt = self.system_prompt_manager.render_block(
            "respond", **(self.system_prompt_manager._prompt_variables | kwargs)
        )
        llm_result = self._run_llm([Message(role=MessageRole.USER, content=prompt)], config, **kwargs).output["content"]
        if self.streaming.enabled and self.streaming.mode == StreamingMode.ALL:
            return self.stream_content(
                content=llm_result, step="manager_response", source=self.name, config=config, **kwargs
            )
        return llm_result
