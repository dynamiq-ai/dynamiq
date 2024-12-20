from abc import ABC, abstractmethod
from typing import ClassVar

from pydantic import BaseModel, Field

from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.agents.base import AgentManager
from dynamiq.nodes.node import NodeDependency, ensure_config
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.utils.logger import logger


class OrchestratorError(Exception):
    """Base exception for Orchestrator errors."""

    pass


class ActionParseError(OrchestratorError):
    """Exception raised when an error occurs during action parsing."""

    pass


class OrchestratorInputSchema(BaseModel):
    input: str = Field(default="", description="The main objective of the orchestration.")


class Orchestrator(Node, ABC):
    """
    Orchestrates the execution of complex tasks using multiple specialized agents.

    Attributes:
        manager (ManagerAgent): The managing agent responsible for overseeing the orchestration process.
        objective (Optional[str]): The main objective of the orchestration.
    """

    name: str | None = "Orchestrator"
    group: NodeGroup = NodeGroup.AGENTS
    input_schema: ClassVar[type[OrchestratorInputSchema]] = OrchestratorInputSchema
    manager: AgentManager
    objective: str = ""

    def __init__(self, **kwargs):
        """
        Initialize the orchestrator with given parameters.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(**kwargs)
        self._run_depends = []
        self._chat_history = []

    def get_final_result(
        self,
        input_data: dict[str, str],
        config: RunnableConfig = None,
        **kwargs,
    ) -> str:
        """
        Generate a comprehensive final result based on the provided data.

        Args:
            input_data (dict[str, str]): Input data for the manager.
            config (RunnableConfig): Configuration for the runnable.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The final comprehensive result.

        Raises:
            OrchestratorError: If an error occurs while generating the final answer.
        """
        logger.debug(f"Orchestrator {self.name} - {self.id}: Running final summarizer")
        manager_result = self.manager.run(
            input_data={"action": "final", **input_data},
            config=config,
            run_depends=self._run_depends,
            **kwargs,
        )
        self._run_depends = [NodeDependency(node=self.manager).to_dict()]

        if manager_result.status != RunnableStatus.SUCCESS:
            error_message = f"Manager '{self.manager.name}' failed: {manager_result.output.get('content')}"
            logger.error(f"Orchestrator {self.name} - {self.id}: Error generating final, due to error: {error_message}")
            raise OrchestratorError(
                f"Orchestrator {self.name} - {self.id}: Error generating final, due to error: {error_message}"
            )

        return manager_result.output.get("content").get("result")

    def reset_run_state(self):
        self._run_depends = []
        self._chat_history = []

    @abstractmethod
    def run_flow(self, input_task: str, config: RunnableConfig = None, **kwargs) -> str:
        """
        Process the given task.

        Args:
            input_task (str): The task to be processed.
            config (RunnableConfig): Configuration for the runnable.

        Returns:
            str: The final answer generated after processing the task.
        """
        pass

    @abstractmethod
    def setup_streaming(self) -> None:
        """Setups streaming for orchestrator."""
        pass

    def execute(self, input_data: OrchestratorInputSchema, config: RunnableConfig = None, **kwargs) -> dict:
        """
        Execute orchestrator flow.

        Args:
            input_data (OrchestratorInputSchema): The input data containing the objective.
            config (Optional[RunnableConfig]): Configuration for the runnable.
            **kwargs: Additional keyword arguments.

        Returns:
            dict[str, Any]: The result of the orchestration process.
        """
        logger.info(f"Orchestrator {self.name} - {self.id}: started with INPUT DATA:\n{input_data}")
        self.reset_run_state()
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        input_task = input_data.input or self.objective

        kwargs = kwargs | {"parent_run_id": kwargs.get("run_id")}
        kwargs.pop("run_depends", None)

        if self.streaming.enabled:
            self.setup_streaming()

        result = self.run_flow(
            input_task=input_task,
            config=config,
            **kwargs,
        )

        logger.info(f"Orchestrator {self.name} - {self.id}: finished with RESULT:\n{str(result)[:200]}...")
        return {"content": result}
