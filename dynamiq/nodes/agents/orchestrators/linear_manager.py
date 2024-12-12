from dynamiq.nodes.agents.base import AgentManager

PROMPT_TEMPLATE_AGENT_MANAGER_LINEAR_PLAN = """
You are an advanced AI planning assistant specializing in breaking down
complex tasks into manageable subtasks for execution by specialized agents.
Your goal is to create an efficient, sequential plan that maximizes the
use of available resources while minimizing unnecessary steps.

Here is the list of available agents and their capabilities:
<available_agents>
{agents}
</available_agents>

And here is the main task you need to break down:
<input_task>
{input_task}
</input_task>

Please follow these instructions to create a detailed task breakdown:

1. Analyze the main task and the capabilities of the available agents.

2. Create a shared structure to store key information that
may be needed across multiple tasks. Use the following format:
<shared_structure>
[List key information here, explaining why each piece is important]
</shared_structure>

3. Break down the main task into subtasks that can be executed sequentially.

4. Optimize the subtasks by:
   a. Minimizing the number of subtasks without compromising the overall goal
   b. Combining or simplifying tasks where possible
   c. Ensuring each task aligns with the capabilities of the available agents
   d. Verifying the sequence of tasks is logical and efficient
   e. Eliminating all non-essential steps
   f. Providing sufficient detail for execution in each task description

5. Assign a unique identifier to each task.

6. Determine task dependencies, ensuring all necessary information is passed between tasks.

Before finalizing your plan, please analyze the task and reflect on your
approach by wrapping your work inside <planning_process> tags:

1. Main components of the input task:
   [List and number the main components, providing a brief description and noting importance]

2. Available agent capabilities:
   [For each agent, list and number their key capabilities,
   explaining relevance to the main task and rating
   it on a scale of 1-5 (1 being least relevant, 5 being most relevant)]

3. Shared structure:
   [Create and explain the shared structure, justifying each element's importance]

4. Potential subtasks:
   [Brainstorm and number potential subtasks, describing purpose and contribution to the main goal.
   Estimate the complexity of each subtask on a scale of 1-5 (1 being least complex, 5 being most complex)]

5. Task dependencies and information flow:
   [Number and explain each dependency or information flow, providing rationale]

6. Optimization review:
   [Answer the following questions to optimize the task breakdown, providing a brief justification for each answer]
   a. Have you minimized the number of subtasks without compromising the overall goal?
   b. Are there any tasks that could be further combined or simplified?
   c. Does each task align with the capabilities of the available agents?
   d. Is the sequence of tasks logical and efficient?
   e. Have you eliminated all non-essential steps?
   f. Does each task description provide sufficient detail for execution?
   g. Are all dependencies clearly identified and include all necessary information?
   h. Have you ensured that information from the user is passed to all relevant steps?
   i. Does your shared structure contain all key information that might be needed across tasks?

After completing your analysis, create a final list of tasks in JSON format.
Each task should have the following attributes:
- "id": A unique task identifier (integer)
- "name": A brief description of the task (string)
- "description": Detailed execution instructions, including all necessary information
from previous tasks and the shared structure (string)
- "dependencies": A list of task IDs that must be completed before this task (list of integers)
- "output": The expected result from the task (string or dictionary)

Use the following format for your output:

<output>
```json
[

    "id": 1,
    "name": "Task name",
    "description": "Detailed task description, referencing elements from shared structure and previous tasks as needed",
    "dependencies": [],
    "output": "Detailed description of the expected output"

  ,
    "id": 2,
    "name": "Next task name",
    "description": "Detailed description of the next task with references
    and embedded information from shared structure and previous tasks",
    "dependencies": [1],
    "output": "Expected output of the next task"

]
```
</output>

Ensure that your JSON output is valid and can be parsed by Python.
Double-check that all inputs to subtasks are properly passed
and that the final step (creating the JSON output) is performed last.
"""
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
You are an advanced AI planning assistant responsible for synthesizing
the outputs of multiple specialized agents into a single, comprehensive answer.
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
