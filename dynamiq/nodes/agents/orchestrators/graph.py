import json
from typing import Any, Callable

from dynamiq.callbacks import NodeCallbackHandler
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.agents.base import Agent
from dynamiq.nodes.agents.orchestrators.graph_manager import GraphAgentManager
from dynamiq.nodes.agents.orchestrators.graph_state import GraphState
from dynamiq.nodes.agents.orchestrators.orchestrator import Orchestrator, OrchestratorError
from dynamiq.nodes.node import Node, NodeDependency
from dynamiq.nodes.tools import Python
from dynamiq.nodes.tools.function_tool import function_tool
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.utils.logger import logger


class StateNotFoundError(OrchestratorError):
    """Raised when next state was not found."""
    pass


START = "START"
END = "END"


class GraphOrchestrator(Orchestrator):
    """
    Orchestrates the execution of complex tasks, interconnected within the graph structure.

    This class manages the execution by following structure of directed graph. When finished synthesizes the results
    into a final answer.

    Attributes:
        manager (ManagerAgent): The managing agent responsible for overseeing the orchestration process.
        context (Dict[str, Any]): Context of the orchestrator.
        states (List[GraphState]): List of states within orchestrator.
        initial_state (str): State to start from.
        objective (Optional[str]): The main objective of the orchestration.
        max_loops (Optional[int]): Maximum number of transition between states.
    """

    name: str | None = "GraphOrchestrator"
    manager: GraphAgentManager
    initial_state: str = START
    context: dict[str, Any] = {}
    states: list[GraphState] = []
    max_loops: int = 15

    def init_components(self, connection_manager: ConnectionManager | None = None) -> None:
        """
        Initialize components of the orchestrator.

        Args:
            connection_manager (Optional[ConnectionManager]): The connection manager. Defaults to ConnectionManager.
        """
        super().init_components(connection_manager)

        if self.manager.is_postponed_component_init:
            self.manager.init_components(connection_manager)

        for state in self.states:
            if state.is_postponed_component_init:
                state.init_components(connection_manager)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._state_by_id = {state.id: state for state in self.states}

        if START not in self._state_by_id:
            start_state = GraphState(id=START, description="Initial state")
            self._state_by_id[START] = start_state
            self.states.append(start_state)

        if END not in self._state_by_id:
            end_state = GraphState(id=END, description="Final state")
            self._state_by_id[END] = end_state
            self.states.append(end_state)

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {"manager": True, "states": True}

    def to_dict(self, **kwargs) -> dict:
        """Converts the instance to a dictionary.

        Returns:
            dict: A dictionary representation of the instance.
        """
        data = super().to_dict(**kwargs)
        data["manager"] = self.manager.to_dict(**kwargs)
        data["states"] = [state.to_dict(**kwargs) for state in self.states]
        return data

    def add_state_by_tasks(
        self, state_id: str, tasks: list[Node | Callable], callbacks: list[NodeCallbackHandler] = []
    ) -> None:
        """
        Adds state to the graph based on tasks.

        Args:
            state_id (str): Id of the state.
            tasks (list[Node | Callable]): List of tasks that have to be executed when running this state.
            callbacks: list[NodeCallbackHandler]: List of callbacks.
        Raises:
            ValueError: If state with specified id already exists.
        """
        if state_id in self._state_by_id:
            raise ValueError(f"Error: State with id {state_id} already exists.")

        filtered_tasks = []

        has_agent = False
        for task in tasks:
            if isinstance(task, Node):
                if isinstance(task, Agent):
                    has_agent = True
                filtered_tasks.append(task)
            elif isinstance(task, Callable):
                filtered_tasks.append(function_tool(task)())
            else:
                raise OrchestratorError("Error: Task must be either a Node or a Callable.")

        state = GraphState(
            id=state_id,
            name=state_id,
            manager=self.manager if has_agent else None,
            tasks=filtered_tasks,
            callbacks=callbacks,
        )
        self.states.append(state)
        self._state_by_id[state.id] = state

    def add_state(self, state: GraphState) -> None:
        """
        Adds state to the graph.

        Args:
            state (State): State to add to the graph.

        Raises:
            ValueError: If state with specified id already exists.
        """
        if state.id in self._state_by_id:
            raise ValueError(f"Error: State with id {state.id} already exists.")

        self.states.append(state)
        self._state_by_id[state.id] = state

    def add_edge(self, source_id: str, destination_id: str) -> None:
        """
        Adds edge to the graph. When source state finishes execution, destination state will be executed next.

        Args:
            source_id (str): Id of source state.
            destination_id (str): Id of destination state.

        Raises:
            ValueError: If state with specified id does not exist.
        """
        self.validate_states([source_id, destination_id])
        self._state_by_id[source_id].next_states = [destination_id]

    def validate_states(self, ids: list[str]) -> None:
        """
        Check if the provided state ids are valid.

        Args:
            ids (list[str]): State ids to validate.

        Raises:
            ValueError: If state with specified id does not exist.
        """
        for state_id in ids:
            if state_id not in self._state_by_id:
                raise ValueError(f"State with id {state_id} does not exist")

    def add_conditional_edge(
        self,
        source_id: str,
        destination_ids: list[str],
        condition: Callable | Python,
        callbacks: list[NodeCallbackHandler] = [],
    ) -> None:
        """
        Adds conditional edge to the graph.
        Conditional edge provides opportunity to choose between destination states based on condition.

        Args:
            source_id (str): Id of the source state.
            destination_ids (list[str]): Ids of destination states.
            condition (Callable | Python): Condition that will determine next state.
            callbacks: list[NodeCallbackHandler]: List of callbacks.
        Raises:
            ValueError: If state with specified id is not present.
        """
        self.validate_states(destination_ids + [source_id])

        if isinstance(condition, Python):
            condition.callbacks.extend(callbacks)
            self._state_by_id[source_id].condition = condition
        elif isinstance(condition, Callable):
            tool = function_tool(condition)()
            tool.callbacks = callbacks
            self._state_by_id[source_id].condition = tool
        else:
            raise OrchestratorError("Error: Conditional edge must be either a Python Node or a Callable.")

        self._state_by_id[source_id].next_states = destination_ids

    def get_next_state_by_manager(self, state: GraphState, config: RunnableConfig, **kwargs) -> GraphState:
        """
        Determine the next state based on the current state and history. Uses GraphAgentManager.

        Args:
            state (State): Current state.
            config (Optional[RunnableConfig]): Configuration for the runnable.
            **kwargs: Additional keyword arguments.

        Returns:
            State: Next state to execute.

        Raises:
            OrchestratorError: If there is an error parsing the action from the LLM response.
            StateNotFoundError: If the state is invalid or not found.
        """
        manager_result = self.manager.run(
            input_data={
                "action": "plan",
                "states_description": self.states_descriptions(state.next_states),
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

        if next_state in self._state_by_id:
            return self._state_by_id[next_state]
        else:
            logger.error(f"GraphOrchestrator: State with id {next_state} was not found.")
            raise StateNotFoundError(f"State with id {next_state} was not found.")

    def _get_next_state(self, state: GraphState, config: RunnableConfig = None, **kwargs) -> GraphState:
        """
        Determine the next state based on the current state and chat history.

        Returns:
            state (State): Current state.

        Raises:
            OrchestratorError: If there is an error parsing output of conditional edge.
            StateNotFoundError: If the state is invalid or not found.
        """
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self._chat_history])

        logger.debug(f"GraphOrchestrator {self.id}: PROMPT {prompt}")

        if len(state.next_states) > 1:
            if condition := state.condition:
                if isinstance(condition, Python):
                    input_data = {**self.context, "history": self._chat_history}
                else:
                    input_data = {"context": self.context | {"history": self._chat_history}}

                next_state = condition.run(
                    input_data=input_data, config=config, run_depends=self._run_depends, **kwargs
                ).output.get("content")

                self._run_depends = [NodeDependency(node=condition).to_dict()]

                if not isinstance(next_state, str):
                    raise OrchestratorError(
                        f"Error: Condition return invalid type. Expected a string got {type(next_state)} "
                    )

                if next_state not in self._state_by_id:
                    raise StateNotFoundError(f"State with id {next_state} was not found.")

                return self._state_by_id[next_state]
            else:
                return self.get_next_state_by_manager(state, config)
        else:
            return self._state_by_id[state.next_states[0]]

    def states_descriptions(self, states: list[str]) -> str:
        """Get a formatted string of states descriptions."""
        return "\n".join(
            [f"'{self._state_by_id[state].name}': {self._state_by_id[state].description}" for state in states]
        )

    def run_flow(self, input_task: str, config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        """
        Process the graph workflow.

        Args:
            input_task (str): The task to be processed.
            config (RunnableConfig): Configuration for the runnable.

        Returns:
            dict[str, Any]: The final output generated after processing the task and inner context of orchestrator.
        """

        self._chat_history.append({"role": "user", "content": input_task})
        state = self._state_by_id[self.initial_state]

        for _ in range(self.max_loops):
            logger.info(f"GraphOrchestrator {self.id}: Next state: {state.id}")

            if state.id == END:
                final_output = self.get_final_result(
                    {
                        "input_task": input_task,
                        "chat_history": self._chat_history,
                    },
                    config=config,
                    **kwargs,
                )
                return {"content": final_output, "context": self.context}

            elif state.id != START:

                output = state.run(
                    input_data={"context": self.context, "chat_history": self._chat_history},
                    config=config,
                    run_depends=self._run_depends,
                    **kwargs,
                ).output

                self.context = self.context | output["context"]
                self._run_depends = [NodeDependency(node=state).to_dict()]
                self._chat_history = self._chat_history + output["history_messages"]

            state = self._get_next_state(state, config=config, **kwargs)

    def setup_streaming(self) -> None:
        """Setups streaming for orchestrator."""
        self.manager.streaming = self.streaming
        for state in self.states:
            for task in state.tasks:
                if isinstance(task, Agent):
                    task.streaming = self.streaming
