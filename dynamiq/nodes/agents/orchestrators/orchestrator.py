import re
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from lxml import etree as LET  # nosec B410
from pydantic import BaseModel, Field

from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.agents.base import AgentManager
from dynamiq.nodes.node import NodeDependency, ensure_config
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.utils.logger import logger


class OrchestratorError(Exception):
    """Base exception for Orchestrator errors."""


class ActionParseError(OrchestratorError):
    """Exception raised when an error occurs during action parsing."""


class OrchestratorInputSchema(BaseModel):
    input: str = Field(default="", description="The main objective of the orchestration.")


class Orchestrator(Node, ABC):
    """
    Orchestrates the execution of complex tasks using multiple specialized agents.

    This abstract base class provides the framework for orchestrating complex tasks
    through multiple agents. It manages the execution flow and communication between
    different specialized agents.

    Attributes:
        manager (ManagerAgent): The managing agent responsible for overseeing the orchestration process.
        objective (Optional[str]): The main objective of the orchestration.

    Abstract Methods:
        run_flow: Processes the given task and returns the result.
        setup_streaming: Configures streaming functionality for the orchestrator.
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
    def run_flow(self, input_task: str, config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        """
        Process the given task.

        Args:
            input_task (str): The task to be processed.
            config (RunnableConfig): Configuration for the runnable.

        Returns:
            dict[str, Any]: The final output generated after processing the task.
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

        Raises:
            OrchestratorError: If the orchestration process fails.
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

        output = self.run_flow(
            input_task=input_task,
            config=config,
            **kwargs,
        )

        logger.info(f"Orchestrator {self.name} - {self.id}: finished with RESULT:\n{str(output)[:200]}...")
        return output

    def _extract_output_content(self, text: str) -> str:
        """
        Extracts the content of the <output> tag. If a properly closed tag is not found,
        fall back to extracting everything after the first occurrence of <output>.
        """
        match = re.search(r"<output>(.*?)</output>", text, re.DOTALL)
        if match:
            return match.group(1).strip()

        start = text.find("<output>")
        if start != -1:
            fallback_content = text[start + len("<output>") :].strip()
            if fallback_content:
                return fallback_content
        raise ActionParseError("No <output> tags found in the response.")

    def _clean_content(self, content: str) -> LET.Element:
        """
        Clean and parse XML content by removing code block markers and wrapping in a root element.

        Args:
            content (str): The input string containing XML content, possibly with code block markers.

        Returns:
            LET.Element: A parsed XML element tree with the cleaned content wrapped in a root element.

        Note:
            - Removes triple backticks (```) from the content
            - Wraps the content in a <root> element for proper XML parsing
            - Uses a lenient XML parser that attempts to recover from malformed XML
        """
        cleaned_content = content.replace("```", "").strip()
        wrapped_content = f"<root>{cleaned_content}</root>"
        parser = LET.XMLParser(recover=True)  # nosec B320
        return LET.fromstring(wrapped_content, parser=parser)  # nosec B320
