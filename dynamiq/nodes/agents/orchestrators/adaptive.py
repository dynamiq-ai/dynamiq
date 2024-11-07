from pydantic import model_validator

from dynamiq.nodes.agents.base import Agent
from dynamiq.nodes.agents.orchestrators.graph import END, START, GraphOrchestrator, State


class OrchestratorError(Exception):
    """Base exception for AdaptiveOrchestrator errors."""

    pass


class ActionParseError(OrchestratorError):
    """Raised when there's an error parsing the LLM action."""

    pass


class AgentNotFoundError(OrchestratorError):
    """Raised when a specified agent is not found."""

    pass


class AdaptiveOrchestrator(GraphOrchestrator):
    """
    Orchestrates the execution of complex tasks using multiple specialized agents.

    This class manages the breakdown of a main objective into subtasks,
    delegates these subtasks to appropriate agents, and synthesizes the results
    into a final answer.

    Attributes:
        manager (ManagerAgent): The managing agent responsible for overseeing the orchestration process.
        agents (List[BaseAgent]): List of specialized agents available for task execution.
        objective (Optional[str]): The main objective of the orchestration.
    """

    agents: list[Agent]

    @model_validator(mode="after")
    def process_agents(self):
        state_names = [agent.name for agent in self.agents]
        states: dict[str, State] = {}
        for agent in self.agents:
            states[agent.name] = State(name=agent.name, connected=state_names + [END], tasks=[agent])

        states[START] = State(name=START, connected=state_names, description="Initial state")
        states[END] = State(name=END, description="Final state")

        self.states = states
