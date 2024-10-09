from enum import Enum
from typing import Any, Callable

from pydantic import BaseModel

from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.agents.base import Agent
from dynamiq.nodes.agents.orchestrators.graph_manager import GraphAgentManager
from dynamiq.nodes.node import NodeDependency, ensure_config
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.utils.logger import logger


class OrchestratorError(Exception):
    """Base exception for GraphOrchestrator errors."""

    pass


class ActionParseError(OrchestratorError):
    """Raised when there's an error parsing the LLM action."""

    pass


class AgentNotFoundError(OrchestratorError):
    """Raised when a specified agent is not found."""

    pass


class ActionCommand(Enum):
    DELEGATE = "delegate"
    CLARIFY = "clarify"
    FINAL_ANSWER = "final_answer"


class Action(BaseModel):
    command: ActionCommand
    agent: str | None = None
    task: str | None = None
    question: str | None = None
    answer: str | None = None


# class Edge(BaseModel):
#     destination_state: 'State' = None
#     condition: str


class State(BaseModel):
    description: str = ""
    connected: list["State"] = []
    tasks: list[Node] = []
    condition: Callable = None


START_STATE = State(description="Initial state")
END_STATE = State(description="Final state")


START = "START"
END = "END"


class GraphOrchestrator(Node):
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

    name: str | None = "GraphOrchestrator"
    group: NodeGroup = NodeGroup.AGENTS
    manager: GraphAgentManager
    agents: list[Agent] = []
    objective: str | None = None
    initial_state: str = START
    states: dict[str, State] = {START: START_STATE, END: END_STATE}

    def __init__(self, **kwargs):
        """
        Initialize the orchestrator with given parameters.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(**kwargs)
        self._chat_history = []
        self._run_depends = []

    def add_node(self, name: str, tasks: list[Node]) -> None:
        if name in self.states:
            raise ValueError(f"State with name {name} is already present")
        state = State(tasks=tasks)

        self.states[name] = state

    def add_edge(self, name_source: list[str], name_destination: str) -> None:
        if name_source not in self.states:
            raise ValueError(f"State with name {name_source} is not present")

        if name_destination not in self.states:
            raise ValueError(f"State with name {name_destination} is not present")
        self.states[name_source].connected.append(self.states[name_destination])

    def add_conditional_edge(self, name_source: str, name_destination: list[str], path_func: Callable) -> None:
        if name_source not in self.states:
            raise ValueError(f"State with name {name_source} is not present")

        for state in name_destination:
            if state not in self.states:
                raise ValueError(f"State with name {name_destination} is not present")
        self.states[name_source].connected += name_destination

        self.states[name_source].condition = path_func

    # def add_edge(self, name, function: Callable) -> None:
    #     if n
    #     self.states[name].

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {"manager": True, "agents": True}

    def to_dict(self, **kwargs) -> dict:
        """Converts the instance to a dictionary.

        Returns:
            dict: A dictionary representation of the instance.
        """
        data = super().to_dict(**kwargs)
        data["manager"] = self.manager.to_dict(**kwargs)
        data["agents"] = [agent.to_dict(**kwargs) for agent in self.agents]
        return data

    def reset_run_state(self):
        self._chat_history = []
        self._run_depends = []

    def init_components(self, connection_manager: ConnectionManager = ConnectionManager()) -> None:
        """
        Initialize components of the orchestrator.

        Args:
            connection_manager (ConnectionManager, optional): The connection manager. Defaults to ConnectionManager.
        """
        super().init_components(connection_manager)
        if self.manager.is_postponed_component_init:
            self.manager.init_components(connection_manager)

        for agent in self.agents:
            if agent.is_postponed_component_init:
                agent.init_components(connection_manager)

    @property
    def agents_descriptions(self) -> str:
        """Get a formatted string of agent descriptions."""
        return "\n".join([f"{i}. {agent.name}" for i, agent in enumerate(self.agents)]) if self.agents else ""

    def get_next_state(self, state: State, config: RunnableConfig = None, **kwargs) -> Action:
        """
        Determine the next action based on the current state and LLM output.

        Returns:
            Action: The next action to be taken.

        Raises:
            ActionParseError: If there is an error parsing the action from the LLM response.
        """
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self._chat_history])

        logger.debug(f"GraphOrchestrator {self.id}: PROMPT {prompt}")

        if len(state.connected) > 1:

            if state.condition:
                next_state_name = state.condition(str(self._chat_history))
                return self.states[next_state_name]

            pass
        else:
            return state.connected[0]

    def run_task(self, task: str, config: RunnableConfig = None, **kwargs) -> str:
        """
        Process the given task using the manager agent logic.

        Args:
            task (str): The task to be processed.
            config (RunnableConfig): Configuration for the runnable.

        Returns:
            str: The final answer generated after processing the task.
        """
        self._chat_history.append({"role": "user", "content": task})

        state = self.states[self.initial_state]
        while True:

            logger.debug(f"GraphOrchestrator {self.id}: chat history: {self._chat_history}")
            logger.debug(f"GraphOrchestrator {self.id}: Next action: {state.model_dump()}")
            if state == self.states[END]:
                return self.get_final_result(
                    input_task=task,
                    preliminary_answer="",
                    config=config,
                    **kwargs,
                )
            elif state != self.states[START]:
                # actions = self.get_actions(state) Should be added because currently relies on agent capabilities only
                self._handle_state_execution(state=state, config=config, **kwargs)

            state = self.get_next_state(state, config=config, **kwargs)

    def _handle_state_execution(self, state: State, config: RunnableConfig = None, **kwargs) -> None:
        """
        Handle task delegation to a specialized agent.

        Args:
            action (Action): The action containing the delegation command and details.
        """

        for agent in state.tasks:
            result = agent.run(
                input_data={
                    "input": "Here is the history of converastion" + str(self._chat_history) + "Continue with your goal"
                },
                config=config,
                run_depends=self._run_depends,
                **kwargs,
            )
            self._run_depends = [NodeDependency(node=agent).to_dict()]
            if result.status != RunnableStatus.SUCCESS:
                logger.error(f"GraphOrchestrator {self.id}: Error executing Agent {agent.name}")
                raise OrchestratorError(
                    f"Failed to execute Agent {agent.name} with Error: {result.output.get('content')}"
                )

            self._chat_history.append(
                {
                    "role": "system",
                    "content": f"Agent {agent.name} result: {result.output.get('content')}",
                }
            )

    def get_final_result(
        self,
        input_task: str,
        preliminary_answer: str,
        config: RunnableConfig = None,
        **kwargs,
    ) -> str:
        """
        Generate a comprehensive final result based on the task and agent outputs.

        Args:
            input_task (str): The original task given.
            preliminary_answer (str): The preliminary answer generated.
            config (RunnableConfig): Configuration for the runnable.

        Returns:
            str: The final comprehensive result.

        Raises:
            OrchestratorError: If an error occurs while generating the final answer.
        """

        manager_result = self.manager.run(
            input_data={
                "action": "final",
                "input_task": input_task,
                "chat_history": self._chat_history,
                "preliminary_answer": preliminary_answer,
            },
            config=config,
            run_depends=self._run_depends,
            **kwargs,
        )
        self._run_depends = [NodeDependency(node=self.manager).to_dict()]

        if manager_result.status != RunnableStatus.SUCCESS:
            logger.error(f"GraphOrchestrator {self.id}: Error generating final answer")
            raise OrchestratorError("Failed to generate final answer")

        return manager_result.output.get("content").get("result")

    def execute(self, input_data: Any, config: RunnableConfig | None = None, **kwargs) -> dict:
        """
        Execute the orchestration process with the given input data and configuration.

        Args:
            input_data (Any): The input data containing the objective or additional context.
            config (Optional[RunnableConfig]): Configuration for the runnable.
            **kwargs: Additional keyword arguments.

        Returns:
            dict[str, Any]: The result of the orchestration process.

        Raises:
            OrchestratorError: If an error occurs during the orchestration process.
        """
        self.reset_run_state()
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        objective = input_data.get("input") or self.objective
        logger.debug(f"GraphOrchestrator {self.id}: Starting the flow with objective:\n```{objective}```.")

        run_kwargs = kwargs | {"parent_run_id": kwargs.get("run_id")}
        run_kwargs.pop("run_depends", None)

        result = self.run_task(
            task=objective,
            config=config,
            **run_kwargs,
        )

        logger.debug(f"GraphOrchestrator {self.id}: Output collected")
        return {"content": result}
