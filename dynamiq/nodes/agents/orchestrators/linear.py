from typing import Any

from pydantic import BaseModel, model_validator

from dynamiq.nodes.agents.base import Agent
from dynamiq.nodes.agents.orchestrators.graph import END, START, GraphOrchestrator, State


class Task(BaseModel):
    """
    Represents a single task in the LinearOrchestrator system.

    Attributes:
        id (int): Unique identifier for the task.
        name (str): Name of the task.
        description (str): Detailed description of the task.
        dependencies (list[int]): List of task IDs that this task depends on.
        output: dict[str, Any] | str: Expected output of the task.
    """

    id: int
    name: str
    description: str
    dependencies: list[int]
    output: dict[str, Any] | str


class LinearOrchestrator(GraphOrchestrator):
    """
    Manages the execution of tasks by coordinating multiple agents and leveraging LLM (Large Language Model).

    Attributes:
        manager (ManagerAgent): The managing agent responsible for overseeing the orchestration process.
        agents (List[BaseAgent]): List of specialized agents available for task execution.
        input_task (str | None): Initial task input.
        use_summarizer (bool): Indicates if a final summarizer is used.
        summarize_all_answers (bool): Indicates whether to summarize answers to all tasks or use only last one.\
              Will only be applied if use_summarizer is set to True.

    """

    name: str | None = "LinearOrchestrator"
    agents: list[Agent]

    @model_validator(mode="after")
    def process_agents(self):
        state_names = [agent.name for agent in self.agents] + [END]
        states: dict[str, State] = {}

        states[START] = State(name=START, connected=[state_names[0]], description="Initial state")

        for index, agent in enumerate(self.agents):
            states[agent.name] = State(name=agent.name, connected=[state_names[index + 1]], tasks=[agent])

        states[END] = State(name=END, description="Final state")

        self.states = states
