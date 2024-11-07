import json
from typing import Any, Callable
from pydantic import BaseModel
from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.agents.base import Agent
from dynamiq.nodes.agents.orchestrators.graph_manager import GraphAgentManager
from dynamiq.nodes.node import NodeDependency, ensure_config
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.utils.logger import logger


class OrchestratorError(Exception):
    """Base exception for AdaptiveOrchestrator errors."""

    pass


class StateNotFoundError(OrchestratorError):
    """Raised when proposed next state was not found."""

    pass


class State(BaseModel):
    name: str
    description: str = ""
    connected: list["str"] = []
    tasks: list[Node | Callable] = []
    branch: Callable = None


START = "START"
END = "END"


START_STATE = State(name=START, description="Initial state")
END_STATE = State(name=END, description="Final state")


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
    initial_state: str = START
    states: dict[str, State] = {START: START_STATE, END: END_STATE}
    context: dict[str, Any] = {"history": []}
    objective: str | None = None

    def __init__(self, **kwargs):
        """
        Initialize the orchestrator with given parameters.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(**kwargs)
        self._run_depends = []

    def add_node(self, name: str, tasks: list[Agent | Callable]) -> None:
        if name in self.states:
            raise ValueError(f"Error: State with name {name} already exists.")

        state = State(name=name, tasks=tasks)
        self.states[name] = state

    def add_edge(self, source: list[str], destination: str) -> None:
        self.validate_states([source, destination])
        self.states[source].connected.append(destination)

    def validate_states(self, names: list[str]):
        """
        Check if the provided state names is valid (exists in the list of valid states).

        Args:
            names (list[str]): names of states to validate.
        """
        for state_name in names:
            if state_name not in self.states:
                raise ValueError(f"State with name {state_name} is not present")

    def add_conditional_edge(self, source: str, destinations: list[str], path_func: Callable) -> None:
        self.validate_states(destinations + [source])
        self.states[source].connected = destinations
        self.states[source].branch = path_func

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
                "chat_history": self.context["history"],
                "preliminary_answer": preliminary_answer,
            },
            config=config,
            run_depends=self._run_depends,
            **kwargs,
        )
        self._run_depends = [NodeDependency(node=self.manager).to_dict()]

        if manager_result.status != RunnableStatus.SUCCESS:
            logger.error(f"AdaptiveOrchestrator {self.id}: Error generating final answer")
            raise OrchestratorError("Failed to generate final answer")

        return manager_result.output.get("content").get("result")

    def get_next_action(self, state: State, config: RunnableConfig, **kwargs):
        """
        Determine the next action based on the current state and LLM output.

        Returns:
            str: name of next state

        Raises:
            OrchestratorError: If there is an error parsing the action from the LLM response.
            StateNotFoundError: If state was not valid or not found.
        """
        manager_result = self.manager.run(
            input_data={
                "action": "plan",
                "states_description": self.states_descriptions(state.connected),
                "chat_history": self.context["history"],
            },
            config=config,
            run_depends=self._run_depends,
            **kwargs,
        )
        # self._run_depends = [NodeDependency(node=self.manager).to_dict()]

        if manager_result.status != RunnableStatus.SUCCESS:
            logger.error(f"GraphOrchestrator {self.id}: Error generating final answer")
            raise OrchestratorError("Failed to generate final answer")

        try:
            next_state = json.loads(
                manager_result.output.get("content").get("result").replace("json", "").replace("```", "").strip()
            )["state"]
        except Exception as e:
            logger.error("GraphOrchestrator: Error when parsing response about next state.")
            raise OrchestratorError(f"Error when parsing response about next state {e}")

        if next_state in list(self.states.keys()):
            return next_state
        else:
            logger.error(f"GraphOrchestrator: State with name {next_state} was not found.")
            raise StateNotFoundError(f"State with name {next_state} was not found.")

    def reset_run_state(self):
        self._run_depends = []
        self.context["history"] = []

    def get_next_state(self, state: State, config: RunnableConfig = None, **kwargs) -> str:
        """
        Determine the next state based on the current state and chat history.

        Returns:
            state: Next state.
        """
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.context["history"]])

        logger.debug(f"GraphOrchestrator {self.id}: PROMPT {prompt}")

        if len(state.connected) > 1:
            if state.branch:
                next_state_name = state.branch(self.context)
                return self.states[next_state_name]
            return self.states[self.get_next_action(state, config)]
        else:
            return self.states[state.connected[0]]

    def task_description(self, task: Agent):
        return ("Name:", task.name, "Role:", task.role)

    def get_actions(self, state: State, config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        """
        Determine actions for current state.

        Returns:
            Action: The next action to be taken for current state.
        """

        actions = {}
        for agent in state.tasks:
            if isinstance(agent, Agent):
                manager_result = self.manager.run(
                    input_data={
                        "action": "assign",
                        "task": self.task_description(agent),
                        "chat_history": self.context["history"],
                    },
                    config=config,
                    run_depends=self._run_depends,
                    **kwargs,
                )
                # self._run_depends = [NodeDependency(node=self.manager).to_dict()]

                if manager_result.status != RunnableStatus.SUCCESS:
                    logger.error(f"GraphOrchestrator {self.id}: Error generating actions for state.")
                    raise OrchestratorError("Error generating actions for state.")

                try:
                    agent_input = json.loads(
                        manager_result.output.get("content")
                        .get("result")
                        .replace("json", "")
                        .replace("```", "")
                        .strip()
                    )["input"]

                    actions[agent.name] = agent_input

                except Exception as e:
                    logger.error("GraphOrchestrator: Error when parsing response about next state.")
                    raise OrchestratorError(f"Error when parsing response about next state {e}")

        return actions

    def states_descriptions(self, states: list[State]) -> str:
        """Get a formatted string of state descriptions."""
        return "\n".join([f"'{self.states[state].name}': {self.states[state].description}" for state in states])

    def run_task(self, task: str, config: RunnableConfig = None, **kwargs) -> str:
        """
        Process the given task using the manager agent logic.

        Args:
            task (str): The task to be processed.xxfx
            config (RunnableConfig): Configuration for the runnable.

        Returns:
            str: The final answer generated after processing the task.
        """
        self.context["history"].append({"role": "user", "content": task})

        state = self.states[self.initial_state]
        while True:
            logger.debug(f"GraphOrchestrator {self.id}: Next action: {state.model_dump()}")
            if state.name == END:
                return self.get_final_result(
                    input_task=task,
                    preliminary_answer="",
                    config=config,
                    **kwargs,
                )

            elif state.name != START:
                self._handle_state_execution(state, config=config, **kwargs)

            state = self.get_next_state(state, config=config, **kwargs)

    def _handle_state_execution(self, state: State, config: RunnableConfig = None, **kwargs) -> None:
        """
        Handle state execution.

        Args:
            action (Action): The action containing the delegation command and details.
        """

        # with ThreadPoolExecutor() as executor:
        #     futures = [executor.submit(do_something, i) for i in range(10)]

        for task in state.tasks:
            # Create actions where manager will determine input for each agent/tool
            if isinstance(task, Agent):

                manager_result = self.manager.run(
                    input_data={
                        "action": "assign",
                        "task": self.task_description(task),
                        "chat_history": self.context["history"],
                    },
                    config=config,
                    run_depends=self._run_depends,
                    **kwargs,
                )
                # self._run_depends = [NodeDependency(node=self.manager).to_dict()]

                if manager_result.status != RunnableStatus.SUCCESS:
                    logger.error(f"GraphOrchestrator {self.id}: Error generating actions for state.")
                    raise OrchestratorError("Error generating actions for state.")

                try:
                    agent_input = json.loads(
                        manager_result.output.get("content")
                        .get("result")
                        .replace("json", "")
                        .replace("```", "")
                        .strip()
                    )["input"]

                    result = task.run(
                        input_data={"input": agent_input},
                        config=config,
                        run_depends=self._run_depends,
                        **kwargs,
                    )

                    output = result.output.get("content")
                    # self._run_depends = [NodeDependency(node=agent).to_dict()]
                    if result.status != RunnableStatus.SUCCESS:
                        logger.error(f"GraphOrchestrator {self.id}: Error executing Agent {task.name}")
                        raise OrchestratorError(f"Failed to execute Agent {task.name} with Error: {output}")
                except Exception as e:
                    logger.error("GraphOrchestrator: Error when parsing response about next state.")
                    raise OrchestratorError(f"Error when parsing response about next state {e}")

            elif isinstance(task, Callable):
                output = task(self.context, config=config)

            self.context["history"].append(
                {
                    "role": "system",
                    "content": f"Result: {output}",
                }
            )

    def execute(self, input_data: dict[Any], config: RunnableConfig | None = None, **kwargs) -> dict:
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
        self.context = self.context | input_data

        self.reset_run_state()
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        objective = input_data.get("input") or self.objective
        logger.debug(f"AdaptiveOrchestrator {self.id}: Starting the flow with objective:\n```{objective}```")

        run_kwargs = kwargs | {"parent_run_id": kwargs.get("run_id")}
        run_kwargs.pop("run_depends", None)

        result = self.run_task(
            task=objective,
            config=config,
            **run_kwargs,
        )

        logger.debug(f"GraphOrchestrator {self.id}: Output collected")
        return {"content": result}
