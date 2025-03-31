import copy
import json
from typing import Any, ClassVar

from pydantic import BaseModel, Field, model_validator

from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.agents.base import Agent
from dynamiq.nodes.agents.orchestrators.graph_manager import GraphAgentManager
from dynamiq.nodes.agents.orchestrators.orchestrator import OrchestratorError
from dynamiq.nodes.node import Node, NodeDependency
from dynamiq.nodes.tools import Python
from dynamiq.nodes.tools.function_tool import FunctionTool
from dynamiq.nodes.types import NodeGroup
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.utils.logger import logger


class StateInputSchema(BaseModel):
    context: dict[str, Any] = Field(..., description="Previous context objects.")
    chat_history: list[dict[str, Any]] = Field(..., description="Previous chat history.")


class GraphState(Node):
    """Represents single state of graph flow

    Attributes:
        id (str): Unique identifier for the state.
        name (str): Name of the state
        description (str): Description of the state.
        next_states (list[str]): List of adjacent node
        tasks (list[Node]): List of tasks that have to be executed in this state.
        condition (Python | FunctionTool): Condition that determines next state to execute.
        manager (GraphAgentManager): The managing agent responsible for overseeing state execution.
    """

    id: str
    name: str = "State"
    group: NodeGroup = NodeGroup.UTILS
    input_schema: ClassVar[type[StateInputSchema]] = StateInputSchema
    description: str = ""
    next_states: list[str] = []
    tasks: list[Node] = []
    condition: Python | FunctionTool | None = None
    manager: GraphAgentManager | None = None

    @model_validator(mode="after")
    def validate_manager(self):
        for task in self.tasks:
            if isinstance(task, Agent) and not self.manager:
                raise ValueError("Error: Provide manager to state to execute agent tasks.")

        return self

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {"manager": True, "tasks": True}

    def to_dict(self, **kwargs) -> dict:
        """Converts the instance to a dictionary.

        Returns:
            dict: A dictionary representation of the instance.
        """
        data = super().to_dict(**kwargs)
        if self.manager:
            data["manager"] = self.manager.to_dict(**kwargs)
        else:
            data["manager"] = None
        data["tasks"] = [task.to_dict(**kwargs) for task in self.tasks]
        return data

    def merge_contexts(self, context_list: list[dict[str, Any]]) -> dict:
        """
        Merges contexts. Raises error when lossless merging is not possible.

        Args:
            context_list (list[dict[str, Any]]): List of contexts to merge.
        Raises:
            OrchestratorError: If multiple changes of the same context variable are detected.
        """
        merged_dict = {}

        for d in context_list:
            for key, value in d.items():
                if key in merged_dict:
                    if merged_dict[key] != value:
                        raise OrchestratorError(f"Error: multiple changes of context variable {key} are detected.")
                merged_dict[key] = value

        return merged_dict

    def agent_description(self, agent: Agent) -> str:
        """
        Creates agent description.

        Args:
            agent (Agent): Agent for which to provide a description.

        Return:
            str: Description of the agent.
        """
        return f"Name: {agent.name}. Role: {agent.role}"

    def init_components(self, connection_manager: ConnectionManager | None = None) -> None:
        """
        Initialize components of the orchestrator.

        Args:
            connection_manager (Optional[ConnectionManager]): The connection manager. Defaults to ConnectionManager.
        """
        super().init_components(connection_manager)
        if self.manager and self.manager.is_postponed_component_init:
            self.manager.init_components(connection_manager)

        for task in self.tasks:
            if task.is_postponed_component_init:
                task.init_components(connection_manager)

    def validate_input_transformer(self, task: Node, input_data: dict[str, Any], **kwargs) -> bool:
        """
        Validates whether input data after transformation is a correct input for the task.

        Args:
            task (Node): Task that have to be executed.
            input_data (dict[str, Any]): Original input to the task.

        Return:
            bool: Whether input data is correct.
        """

        if task.input_transformer:
            try:
                output = task.transform(input_data, task.input_transformer, task.id)
                task.validate_input_schema(output, **kwargs)

                return True

            except Exception:
                return False
        else:
            return False

    def _submit_task(
        self,
        task: Node,
        global_context: dict[str, Any],
        chat_history: list[dict[str, str]],
        config: RunnableConfig,
        **kwargs,
    ) -> tuple[str, dict[str, Any]]:
        """Executes single task.

        Args:
            task (Node): Task to be executed.
            global_context (dict[str, Any]): Current context of the execution.
            chat_history (list[dict[str, str]]): List of history messages.
            config (Optional[RunnableConfig]): Configuration for the runnable.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The result of the task execution.
            dict[str, Any]: Updates to the context.

        Raises:
            OrchestratorError: If an error occurs during the execution process.

        """

        run_depends = []

        if isinstance(task, Agent):
            agent_input = {"context": global_context | {"history": chat_history}}
            is_input_correct = self.validate_input_transformer(task, agent_input, **kwargs)
            if not is_input_correct:
                manager_result = self.manager.run(
                    input_data={
                        "action": "assign",
                        "task": self.agent_description(task),
                        "chat_history": chat_history,
                    },
                    run_depends=run_depends,
                    config=config,
                    **kwargs,
                )

                run_depends = [NodeDependency(node=self.manager).to_dict()]

                if manager_result.status != RunnableStatus.SUCCESS:
                    logger.error(f"GraphOrchestrator: Error generating actions for state: {manager_result}")
                    raise OrchestratorError(f"GraphOrchestrator: Error generating actions for state: {manager_result}")

                try:
                    agent_input = {
                        "input": json.loads(
                            manager_result.output.get("content")
                            .get("result")
                            .replace("json", "")
                            .replace("```", "")
                            .strip()
                        )["input"]
                    }
                except Exception as e:
                    logger.error(f"GraphOrchestrator: Error when parsing response about next state {e}")
                    raise OrchestratorError(f"Error when parsing response about next state {e}")

            response = task.run(
                input_data=agent_input,
                config=config,
                run_depends=run_depends,
                use_input_transformer=is_input_correct,
                **kwargs,
            )

            result = response.output.get("content")

            if response.status != RunnableStatus.SUCCESS:
                logger.error(f"GraphOrchestrator: Failed to execute Agent {task.name} with Error: {result}")
                raise OrchestratorError(f"Failed to execute Agent {task.name} with Error: {result}")

            return result, {}

        elif isinstance(task, FunctionTool):
            input_data = {"context": global_context | {"history": chat_history}}
        else:
            input_data = {**global_context, "history": chat_history}

        response = task.run(input_data=input_data, config=config, run_depends=run_depends, **kwargs)

        context = response.output.get("content")

        if response.status != RunnableStatus.SUCCESS:
            logger.error(f"GraphOrchestrator: Failed to execute {task.name} with Error: {context}")
            raise OrchestratorError(f"Failed to execute {task.name} with Error: {context}")

        if not isinstance(context, dict):
            raise OrchestratorError(
                f"Error: Task returned invalid data format. Expected a dictionary got {type(context)}"
            )

        if "result" not in context:
            raise OrchestratorError("Error: Task returned dictionary with no 'result' key in it.")

        context.pop("history", None)

        result = context.pop("result")

        return result, context

    def execute(self, input_data: StateInputSchema, config: RunnableConfig = None, **kwargs) -> dict:
        """
        Execute the State.

        Args:
            input_data (StateInputSchema): The input data containing context and chat history.
            config (Optional[RunnableConfig]): Configuration for the runnable.
            **kwargs: Additional keyword arguments.

        Returns:
            dict[str, Any]: The result of the state execution (updated context and chat history).
        """
        logger.debug(f"State {self.id}: starting the flow with input_task:\n```{input_data}```")
        kwargs.pop("run_depends", None)
        self.run_on_node_execute_run(config.callbacks, **kwargs)
        kwargs = kwargs | {"parent_run_id": kwargs.get("run_id")}

        global_context = input_data.context
        chat_history = input_data.chat_history

        history_messages = []

        if len(self.tasks) == 1:
            result, context = self._submit_task(
                self.tasks[0],
                global_context,
                chat_history,
                config=config,
                **kwargs,
            )

            history_messages.append(
                {
                    "role": "system",
                    "content": f"Result: {result}",
                }
            )

            global_context = global_context | context

        elif len(self.tasks) > 1:

            contexts = []

            for task in self.tasks:

                result, context = self._submit_task(
                    task, copy.deepcopy(global_context), copy.deepcopy(chat_history), config=config, **kwargs
                )

                history_messages.append(
                    {
                        "role": "system",
                        "content": f"Result: {result}",
                    }
                )

                contexts.append(context)
            global_context = global_context | self.merge_contexts(contexts)

        return {"context": global_context, "history_messages": history_messages}
