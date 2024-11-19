import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from pydantic import BaseModel

from dynamiq.nodes.agents.base import Agent
from dynamiq.nodes.agents.orchestrators.graph_manager import GraphAgentManager
from dynamiq.nodes.agents.orchestrators.orchestrator import Orchestrator, OrchestratorError
from dynamiq.nodes.node import Node, NodeDependency
from dynamiq.nodes.tools import Python
from dynamiq.nodes.tools.function_tool import FunctionTool, function_tool
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.utils.logger import logger


class StateNotFoundError(OrchestratorError):
    """Raised when next state was not found."""
    pass


class State(BaseModel):
    """Represents single state of graph flow

    Attributes:
        name (str): Name of the state
        description (str): Description of the state.
        adjacent_states (list[str]): List of adjacent node
        tasks (list[Node]): List of tasks that have to be executed in this state.
        conditional_edge (Python | FunctionTool): Condition that determines next state to execute.
    """
    name: str
    description: str = ""
    adjacent_states: list[str] = []
    tasks: list[Node] = []
    conditional_edge: Python | FunctionTool = None


START = "START"
END = "END"


START_STATE = State(name=START, description="Initial state")
END_STATE = State(name=END, description="Final state")


class GraphOrchestrator(Orchestrator):
    """
    Orchestrates the execution of complex tasks using multiple specialized agents.

    This class manages the breakdown of a main objective into subtasks,
    delegates these subtasks to appropriate agents, and synthesizes the results
    into a final answer.

    Attributes:
        manager (ManagerAgent): The managing agent responsible for overseeing the orchestration process.
        agents (List[BaseAgent]): List of specialized agents available for task execution.
        objective (Optional[str]): The main objective of the orchestration.
        max_loops (Optional[int]): Maximum number of transition between states.
    """

    name: str | None = "GraphOrchestrator"
    manager: GraphAgentManager
    initial_state: str = START
    states: dict[str, State] = {START: START_STATE, END: END_STATE}
    context: dict[str, Any] = {}
    max_loops: int = 15

    def add_node(self, name: str, tasks: list[Node | Callable]) -> None:
        """
        Adds state with specified tasks to the graph.

        Args:
            name (str): Name of state.
            tasks (list[Agent | Callable]): List of tasks that have to be executed when running this state.

        Raises:
            ValueError: If state with specified name already exists.
        """
        if name in self.states:
            raise ValueError(f"Error: State with name {name} already exists.")

        filtered_tasks = []

        for task in tasks:
            if isinstance(task, Node):
                filtered_tasks.append(task)
            elif isinstance(task, Callable):
                filtered_tasks.append(function_tool(task)())
            else:
                raise OrchestratorError("Error: Task must be either a Node or a Callable.")

        state = State(name=name, tasks=filtered_tasks)
        self.states[name] = state

    def add_edge(self, source: str, destination: str) -> None:
        """
        Adds edge to the graph.

        Args:
            source (str): Name of source state.
            destination (str): Name of destinations state.

        Raises:
            ValueError: If state with specified name is not present.
        """
        self.validate_states([source, destination])
        self.states[source].adjacent_states = [destination]

    def validate_states(self, names: list[str]) -> None:
        """
        Check if the provided state names is valid (exists in the list of valid states).

        Args:
            names (list[str]): names of states to validate.

        Raises:
            ValueError: If state with specified name is not present.
        """
        for state_name in names:
            if state_name not in self.states:
                raise ValueError(f"State with name {state_name} is not present")

    def add_conditional_edge(self, source: str, destinations: list[str], condition: Callable | Python) -> None:
        """
        Adds conditional edge to the graph.

        Args:
            source (str): Name of source state.
            destinations (list[str]): Name of destinations states.
            path_func (Callable): Function that will determine next state.

        Raises:
            ValueError: If state with specified name is not present.
        """
        self.validate_states(destinations + [source])

        if isinstance(condition, Python):
            self.states[source].conditional_edge = condition
        elif isinstance(condition, Callable):
            self.states[source].conditional_edge = function_tool(condition)()
        else:
            raise OrchestratorError("Error: Conditional edge must be either a Python Node or a Callable.")

        self.states[source].adjacent_states = destinations

    def get_next_state_by_manager(self, state: State, config: RunnableConfig, **kwargs) -> State:
        """
        Determine the next state based on the current state and history. Uses GraphAgentManager.

        Args:
            state (State): Current state.
            config (Optional[RunnableConfig]): Configuration for the runnable.
            **kwargs: Additional keyword arguments.

        Returns:
            State: Next state.

        Raises:
            OrchestratorError: If there is an error parsing the action from the LLM response.
            StateNotFoundError: If state is invalid or not found.
        """
        manager_result = self.manager.run(
            input_data={
                "action": "plan",
                "states_description": self.states_descriptions(state.adjacent_states),
                "chat_history": self._chat_history,
            },
            config=config,
            run_depends=self._run_depends,
            **kwargs,
        )
        self._run_depends = [NodeDependency(node=self.manager).to_dict()]

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

        if next_state in self.states:
            return self.states[next_state]
        else:
            logger.error(f"GraphOrchestrator: State with name {next_state} was not found.")
            raise StateNotFoundError(f"State with name {next_state} was not found.")

    def _get_next_state(self, state: State, config: RunnableConfig = None, **kwargs) -> str:
        """
        Determine the next state based on the current state and chat history.

        Returns:
            state (State): Current state.

        Raises:
            OrchestratorError: If there is an error parsing output of conditional edge.
            StateNotFoundError: If state is invalid or not found.
        """
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self._chat_history])

        logger.debug(f"GraphOrchestrator {self.id}: PROMPT {prompt}")

        if len(state.adjacent_states) > 1:
            if condition := state.conditional_edge:
                if isinstance(condition, Python):
                    input_data = {**self.context, "history": self._chat_history}
                else:
                    input_data = {"context": self.context | {"history": self._chat_history}}

                next_state = condition.run(
                    input_data=input_data,
                    config=config,
                    run_depends=self._run_depends,
                ).output.get("content")

                if not isinstance(next_state, str):
                    raise OrchestratorError(
                        f"Error: Condition return invalid type. Expected a string got {type(next_state)} "
                    )

                if next_state not in self.states:
                    raise StateNotFoundError(f"State with name {next_state} was not found.")

                return self.states[next_state]
            else:
                return self.get_next_state_by_manager(state, config)
        else:
            return self.states[state.adjacent_states[0]]

    def task_description(self, task: Agent):
        return ("Name:", task.name, "Role:", task.role)

    def states_descriptions(self, states: list[State]) -> str:
        """Get a formatted string of state descriptions."""
        return "\n".join([f"'{self.states[state].name}': {self.states[state].description}" for state in states])

    def run_flow(self, input_task: str, config: RunnableConfig = None, **kwargs) -> str:
        """
        Process the graph workflow.

        Args:
            input_task (str): The task to be processed.
            config (RunnableConfig): Configuration for the runnable.

        Returns:
            str: The final answer generated after processing the task.
        """
        self._chat_history.append({"role": "user", "content": input_task})
        state = self.states[self.initial_state]

        for _ in range(self.max_loops):
            logger.info(f"GraphOrchestrator {self.id}: Next state: {state.name}")

            if state.name == END:
                return self.get_final_result(
                    {
                        "input_task": input_task,
                        "chat_history": self._chat_history,
                    },
                    config=config,
                    **kwargs,
                )

            elif state.name != START:
                self._handle_state_execution(state, config=config, **kwargs)

            state = self._get_next_state(state, config=config, **kwargs)

    def _submit_task(self, task, config, **kwargs) -> str:
        """Executes task

        Args:
            task (Agent | FunctionTool): Task to be executed.
            config (Optional[RunnableConfig]): Configuration for the runnable.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The result of the task execution.

        Raises:
            OrchestratorError: If an error occurs during the execution process.

        """
        if isinstance(task, Agent):

            thread_run_depends = self._run_depends
            manager_result = self.manager.run(
                input_data={
                    "action": "assign",
                    "task": self.task_description(task),
                    "chat_history": self._chat_history,
                },
                config=config,
                run_depends=thread_run_depends,
                **kwargs,
            )

            thread_run_depends = [NodeDependency(node=self.manager).to_dict()]

            if manager_result.status != RunnableStatus.SUCCESS:
                logger.error("GraphOrchestrator: Error generating actions for state.")
                raise OrchestratorError("Error generating actions for state.")

            try:
                agent_input = json.loads(
                    manager_result.output.get("content").get("result").replace("json", "").replace("```", "").strip()
                )["input"]

                result = task.run(
                    input_data={"input": agent_input},
                    config=config,
                    run_depends=thread_run_depends,
                    **kwargs,
                )

                content = result.output.get("content")

                if result.status != RunnableStatus.SUCCESS:
                    logger.error(f"GraphOrchestrator: Failed to execute Agent {task.name} with Error: {content}")
                    raise OrchestratorError(f"Failed to execute Agent {task.name} with Error: {content}")

                return content, {}

            except Exception as e:
                logger.error(f"GraphOrchestrator: Error when parsing response about next state {e}")
                raise OrchestratorError(f"Error when parsing response about next state {e}")

        elif isinstance(task, FunctionTool):
            result = task.run(
                input_data={"context": self.context | {"history": self._chat_history}},
                config=config,
                run_depends=self._run_depends,
            ).output.get("content")

        elif isinstance(task, Python):
            result = task.run(
                input_data={**self.context, "history": self._chat_history}, config=config, run_depends=self._run_depends
            ).output.get("content")

        if not isinstance(result, dict):
            raise OrchestratorError(
                f"Error: Task returned invalid data format. Expected a dictionary got {type(result)}"
            )

        if "result" not in result:
            raise OrchestratorError("Error: Task returned dictionary with no 'result' key in it.")

        if "history" in result:
            result.pop("history")

        content = result.pop("result")

        return content, result

    def merge_contexts(self, context_list: list[dict[str, Any]]) -> dict:
        """
        Merges contexts, and raises an error when lossless merging is not possible.

        Raises:
            OrchestratorError: If multiple changes of the same variable are detected.
        """
        merged_dict = {}

        for d in context_list:
            for key, value in d.items():
                if key in merged_dict:
                    if merged_dict[key] != value:
                        raise OrchestratorError(f"Error: multiple changes of variable {key} are detected.")
                merged_dict[key] = value

        return merged_dict

    def _handle_state_execution(self, state: State, config: RunnableConfig = None, **kwargs) -> None:
        """
        Handle state execution.

        Args:
            action (Action): The action containing the delegation command and details.
            config (Optional[RunnableConfig]): Configuration for the runnable.
            **kwargs: Additional keyword arguments.
        """
        if len(state.tasks) == 1:
            result, context = self._submit_task(state.tasks[0], config=config, **kwargs)

            self._chat_history.append(
                {
                    "role": "system",
                    "content": f"Result: {result}",
                }
            )

            self.context = self.context | context

        elif len(state.tasks) > 1:

            contexts = []

            with ThreadPoolExecutor() as executor:

                futures = [executor.submit(self._submit_task, task, config=config, **kwargs) for task in state.tasks]

                for future in futures:

                    result, context = future.result()

                    self._chat_history.append(
                        {
                            "role": "system",
                            "content": f"Result: {result}",
                        }
                    )

                    contexts.append(context)

            self.context = self.context | self.merge_contexts(contexts)

        self._run_depends = [NodeDependency(node=task).to_dict() for task in state.tasks]

    def setup_streaming(self) -> None:
        """Setups streaming for orchestrator."""
        self.manager.streaming = self.streaming
        for state in self.states:
            for task in state.tasks:
                if isinstance(task, Agent):
                    task.streaming = self.streaming
