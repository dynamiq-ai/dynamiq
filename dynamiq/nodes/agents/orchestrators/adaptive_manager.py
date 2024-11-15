from dynamiq.nodes.agents.base import AgentManager

PROMPT_TEMPLATE_AGENT_MANAGER_PLAN = """
You are the Manager Agent responsible for coordinating a team of specialized agents to complete complex tasks.
Your role is to:
1. Understand the overall task
2. Break down the task into subtasks
3. Delegate subtasks to appropriate specialized agents
4. Synthesize results from specialized agents
5. Request clarification from the user when needed (If appropriate tools are available)
6. Provide a final answer when the task is complete


Available specialized agents:
{agents}

Respond with a JSON object representing your next action. Use one of the following formats:

For delegation:
"command": "delegate", "agent": "<agent_name>", "task": "<task_description>"

For final answer:
"command": "final_answer", "answer": "<your_final_answer>"

Always think step-by-step and consider the most efficient way to complete the given task.
Provide your response in JSON format only, without any additional text.
For "final_answer" command the final answer should be provided as the value for the 'answer' key. Keep the text within the 'answer' unchanged.
{chat_history}
"""  # noqa: E501

PROMPT_TEMPLATE_AGENT_MANAGER_FINAL_ANSWER = """
Original Task: {input_task}

You have completed the task using various specialized agents. Here's a summary of the work done:

{chat_history}

Preliminary Answer: {preliminary_answer}

Your task now is to provide a comprehensive and coherent final answer to the original task.
This answer should:
1. Directly address the original task
2. Synthesize the information gathered by all agents
3. Present a clear, well-structured response
4. Include relevant details and insights
5. Be written in a professional tone suitable for a final report

Please generate the final answer:
"""  # noqa: E501


class AdaptiveAgentManager(AgentManager):
    """An adaptive agent manager that coordinates specialized agents to complete complex tasks."""

    name: str = "Adaptive Manager"

    def __init__(self, **kwargs):
        """Initialize the AdaptiveAgentManager and set up prompt templates."""
        super().__init__(**kwargs)
        self._init_prompt_blocks()

    def _init_prompt_blocks(self):
        """Initialize the prompt blocks with adaptive plan and final prompts."""
        super()._init_prompt_blocks()
        self._prompt_blocks.update(
            {
                "plan": self._get_adaptive_plan_prompt(),
                "final": self._get_adaptive_final_prompt(),
            }
        )

    @staticmethod
    def _get_adaptive_plan_prompt() -> str:
        """Return the adaptive plan prompt template."""
        return PROMPT_TEMPLATE_AGENT_MANAGER_PLAN

    @staticmethod
    def _get_adaptive_final_prompt() -> str:
        """Return the adaptive final answer prompt template."""
        return PROMPT_TEMPLATE_AGENT_MANAGER_FINAL_ANSWER
