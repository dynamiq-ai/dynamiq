from dynamiq.nodes.agents.base import AgentManager

PROMPT_TEMPLATE_AGENT_MANAGER_LINEAR_PLAN = """
You are an advanced AI planning assistant tasked with breaking down complex tasks into manageable subtasks
for execution by specialized agents.
Your goal is to create an efficient, sequential plan that maximizes the use of available resources while minimizing unnecessary steps.

Here is the main task you need to break down:

<input_task>
{input_task}
</input_task>

And here is the list of available agents and their capabilities:

<agents>
{agents}
</agents>

Please follow these instructions carefully:

1. Analyze the main task and the capabilities of the available agents.
2. Break down the main task into subtasks that can be executed sequentially.
3. Create as few subtasks as possible while ensuring each is actionable and produces useful outputs for subsequent steps.
4. Unify subtasks that can be executed in one step.
5. Remove any steps that are not obligatory for successful task execution.
6. Ensure each task is simple, with one or two steps, and directly addresses the initial problem.
7. Assign a unique identifier to each task.
8. Determine task dependencies.

Before finalizing your plan, please analyze the task and reflect on your approach in <task_analysis> tags:

1. List the main components of the input task.
2. Note the key capabilities of each available agent.
3. Brainstorm potential subtasks, numbering them as you go. It's OK for this section to be quite long.
4. Consider task dependencies and potential combinations.
5. Review your initial breakdown and optimize it by answering these questions:
   - Have you minimized the number of subtasks without compromising the overall goal?
   - Are there any tasks that could be further combined or simplified?
   - Does each task align with the capabilities of the available agents?
   - Is the sequence of tasks logical and efficient?
   - Have you eliminated all non-essential steps?

After your analysis, create a final list of tasks in JSON format within <output> tag.Each task should have the following attributes:
- "id": A unique task identifier (int)
- "name": A brief description of the task (string)
- "description": Detailed execution instructions (string)
- "dependencies": A list of task IDs that must be completed before this task (list of ints)
- "output": The expected result from the task (string or dictionary)

Here's an example of the expected JSON structure (with generic content):
<output>
```json
[
    "id": "1",
    "name": "Initialize project",
    "description": "Set up the project environment and gather necessary resources.",
    "dependencies": [],
    "output": "Project environment ready for subsequent tasks"
,
    "id": "1",
    "name": "Process data",
    "description": "Clean and normalize the input data for analysis.",
    "dependencies": ["1"],
    "output": "Processed dataset ready for analysis"
]
```
</output>
Please ensure that your XML tags are properly formatted and that your JSON output is valid and can be parsed by Python.

Begin by analyzing the task and reflecting on your approach:
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
You are an advanced AI planning assistant responsible for synthesizing the outputs of multiple specialized agents into a single, comprehensive answer.
Your task is to analyze the results of subtasks and provide a cohesive response to the initial request.

Here is the initial major task that was broken down and assigned to specialized agents:

<input_task>
{input_task}
</input_task>

Below are the outputs from the specialized agents who worked on the subtasks:

<tasks_outputs>
{tasks_outputs}
</tasks_outputs>

Your goal is to craft a final, unified answer based on all the task outputs.
This answer should directly address the initial request and provide a complete response to the user.

Before providing your final answer, break down your thought process inside <task_synthesis> tags. Follow these steps:

1. Summarize the initial task in one or two sentences to ensure you understand the main objective.
2. For each subtask output:
   - List the key information relevant to the initial request.
   - Note any unique insights or perspectives provided.
3. Identify connections between the different subtask outputs, noting how they relate to each other and to the overall task.
4. Outline the most important points that need to be included in the final answer.
5. Propose a structure for your response that will provide a clear and concise answer.

Now, based on your analysis, please provide your final answer in the following structure:
<final_answer>
[Write your final answer here, ensuring it's well-structured and addresses the original task comprehensively.
If user asks for a specific format, please follow that format.
If user asks about some code, reports, or tables, please include them in the final answer.
Include all relevant information to be as informative as possible.
]
</final_answer>

Please proceed with your task synthesis and final answer.
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
