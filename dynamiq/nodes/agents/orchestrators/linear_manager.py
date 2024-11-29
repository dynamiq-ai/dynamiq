from dynamiq.nodes.agents.base import AgentManager

PROMPT_TEMPLATE_AGENT_MANAGER_LINEAR_PLAN = """

You are a helpful planning assistant. Your task is to break down the input task into as few smaller subtasks as possible that can be executed by different specified agents based on their capabilities.

**Planning Instructions:**
- Decompose the main task into subtasks for sequential execution.
- Make as few subtasks as possible.
- Unify subtasks that can be executed in one step.
- Avoid steps that are not obligatory for successful task execution.
- Ensure each task is actionable and produces outputs useful for subsequent steps.
- Tasks should be simple, with one or two steps, and must address the initial problem effectively.

**Planner Output Format:**
- List of tasks with attributes:
  - `id`: Unique task identifier.
  - `name`: Brief description of the task.
  - `description`: Detailed execution instructions.
  - `dependencies`: List of task IDs that must be completed before this task.
  - `output`: Expected result from the task.

**Input task**:
{input_task}

**Agents**:
{agents}

Begin planning by breaking down the task into subtasks. Return the list in JSON format.
Remember JSON must be readable by Python.
"""  # noqa: E501

PROMPT_TEMPLATE_AGENT_MANAGER_LINEAR_ASSIGN = """
You are a helpful agent responsible for recommending the best agent for a specific task.

**Initial major task**
{input_task}

**Current task**
{task}

**Available agents**
{agents}

Consider the agent's role, description, tools, and task dependencies. Return only the selected agent's number.
"""  # noqa: E501

PROMPT_TEMPLATE_AGENT_MANAGER_LINEAR_FINAL_ANSWER = """
You are a helpful agent responsible for providing the final answer to the customer's request.

**Initial major task**
{input_task}

**Tasks Outputs**
{tasks_outputs}

Craft the final answer based on all task outputs. Return only the final answer.
"""  # noqa: E501


PROMPT_TEMPLATE_MANAGER_LINEAR_RUN = """
You are a helpful agent responsible for recommending the best agent for a specific task.

**Current task**
{task}

Provide answer on current task.
"""


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

    def _init_prompt_blocks(self):
        """
        Initializes the prompt blocks used in the task planning, assigning,
        and final answer generation processes.
        """
        super()._init_prompt_blocks()
        self._prompt_blocks.update(
            {
                "plan": self._get_linear_plan_prompt(),
                "assign": self._get_linear_assign_prompt(),
                "final": self._get_linear_final_prompt(),
            }
        )

    @staticmethod
    def _get_linear_plan_prompt() -> str:
        """
        Returns the prompt template for planning tasks.
        """
        return PROMPT_TEMPLATE_AGENT_MANAGER_LINEAR_PLAN

    @staticmethod
    def _get_linear_assign_prompt() -> str:
        """
        Returns the prompt template for assigning tasks to agents.
        """
        return PROMPT_TEMPLATE_AGENT_MANAGER_LINEAR_ASSIGN

    @staticmethod
    def _get_linear_final_prompt() -> str:
        """
        Returns the prompt template for generating the final answer.
        """
        return PROMPT_TEMPLATE_AGENT_MANAGER_LINEAR_FINAL_ANSWER

    @staticmethod
    def _get_linear_agent_run() -> str:
        """
        Returns the prompt template for question answering by Linear Manager.
        """
        return PROMPT_TEMPLATE_MANAGER_LINEAR_RUN
