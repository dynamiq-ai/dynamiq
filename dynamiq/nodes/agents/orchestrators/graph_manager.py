from dynamiq.nodes.agents.base import AgentManager

PROMPT_TEMPLATE_AGENT_MANAGER_NEXT_STATE = """
You are the Manager Agent responsible for coordinating a team of specialized agents to complete complex tasks.
Your role is to delegate to appropriate new state based on description

Available states:
{states_description}

Respond with a JSON object representing your next action. Use one of the following formats:

For choosing next state:
"state": "<state_name>"

Provide your response in JSON format only, without any additional text.
For the final answer this means providing the final answer as the value for the "answer" key. But text in answer keep as it is.
{chat_history}
"""  # noqa: E501

PROMPT_TEMPLATE_AGENT_MANAGER_ACTIONS = """
You are the Manager Agent responsible for coordinating a team of specialized agents to complete complex tasks.
Your role is to generate input query for each of the agents based on previous history

Agent:
{task}

Provide your response in JSON format only, without any additional text.

For providing action:
"command": "action", "agent": "<agent_name>", "input": "<input_to_agent>"

Chat history:
{chat_history}
"""  # noqa: E501


PROMPT_TEMPLATE_AGENT_MANAGER_FINAL_ANSWER = """
Original Task: {input_task}

You have completed the task using various specialized agents. Here's a summary of the work done:

{chat_history}

Your task now is to provide a comprehensive and coherent final answer to the original task.
This answer should:
1. Directly address the original task
2. Synthesize the information gathered by all agents
3. Present a clear, well-structured response
4. Include relevant details and insights
5. Be written in a professional tone suitable for a final report

Please generate the final answer:
"""  # noqa: E501


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
        self._prompt_blocks.update(
            {
                "plan": self._get_next_state_prompt(),
                "assign": self._get_actions_prompt(),
                "final": self._get_adaptive_final_prompt(),
            }
        )

    @staticmethod
    def _get_next_state_prompt() -> str:
        """Return next step prompt template."""
        return PROMPT_TEMPLATE_AGENT_MANAGER_NEXT_STATE

    @staticmethod
    def _get_actions_prompt() -> str:
        """Return actions prompt template."""
        return PROMPT_TEMPLATE_AGENT_MANAGER_ACTIONS

    @staticmethod
    def _get_adaptive_final_prompt() -> str:
        """Return the adaptive final answer prompt template."""
        return PROMPT_TEMPLATE_AGENT_MANAGER_FINAL_ANSWER
