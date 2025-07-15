import json
import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, ClassVar

from lxml import etree as LET  # nosec B410
from pydantic import BaseModel, Field

from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.agents.base import AgentManager
from dynamiq.nodes.node import NodeDependency, ensure_config
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types.streaming import StreamingMode
from dynamiq.utils.logger import logger


class OrchestratorError(Exception):
    """Base exception for Orchestrator errors."""


class ActionParseError(OrchestratorError):
    """Exception raised when an error occurs during action parsing."""


class OrchestratorInputSchema(BaseModel):
    input: str = Field(default="", description="The main objective of the orchestration.")


class Decision(str, Enum):
    """
    Enumeration for possible decisions after analyzing the user input.
    """

    RESPOND = "respond"
    PLAN = "plan"


class DecisionResult(BaseModel):
    """
    Holds the result of analyzing the user input.

    Attributes:
        decision (Decision): The decision on how to handle the input.
        message (str): The message or response associated with the decision.
    """

    decision: Decision
    message: str


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
    enable_handle_input: bool = True

    def __init__(self, **kwargs):
        """
        Initialize the orchestrator with given parameters.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(**kwargs)
        self._run_depends = []
        self._chat_history = []

    def _clean_output(self, text: str) -> str:
        """Remove Markdown code fences and extra whitespace from a text."""
        cleaned = re.sub(r"^```(?:json)?\s*|```$", "", text).strip()
        return cleaned

    def extract_json_from_output(self, result_text: str) -> tuple[str, dict] | None:
        """
        Extracts JSON data from the given text by looking for content within
        <output>...</output> and <analysis>...</analysis> tags. Strips any Markdown code block fences.

        Args:
            result_text (str): The text from which to extract JSON data.

        Returns:
            dict | None: The extracted JSON dictionary if successful, otherwise None.
        """
        analysis, output_content = self._extract_output_content(result_text)
        output_content = self._clean_output(output_content)

        try:
            data = json.loads(output_content)
            return analysis, data
        except json.JSONDecodeError as e:
            error_message = f"Orchestrator {self.name} - {self.id}: JSON decoding error: {e}"
            logger.error(error_message)
            return None

    def _analyze_user_input(
        self, input_task: str, description: str, config: RunnableConfig = None, attempt: int = 1, **kwargs
    ) -> DecisionResult:
        """
        Calls the manager's 'handle_input' action to decide if we should respond
        immediately or proceed with a plan.

        Args:
            input_task (str): The user's input or task description.
            description (str): Description of the orchestrator capabilities.
            config: Optional configuration object passed to the manager's run method.
            attempt (int): The current attempt number for analyzing user input.
            **kwargs: Additional keyword arguments passed to the manager's run method.

        Returns:
            DecisionResult: An object containing the decision (as an Enum) and a message.
        """
        if not self.enable_handle_input:
            return DecisionResult(decision=Decision.PLAN, message="")

        handle_result = self.manager.run(
            input_data={
                "action": "handle_input",
                "task": input_task,
                "description": description,
            },
            config=config,
            run_depends=[],
            **kwargs,
        )

        if handle_result.status != RunnableStatus.SUCCESS:
            error = handle_result.error.to_dict()
            error_message = f"Orchestrator {self.name} - {self.id}: Manager failed to analyze input: {error}"
            logger.error(error_message)
            return DecisionResult(decision=Decision.RESPOND, message=f"Error analyzing request: {error}")

        content = handle_result.output.get("content", {})
        raw_text = content.get("result", "")
        if not raw_text:
            error_message = f"Orchestrator {self.name} - {self.id}: No 'result' field in manager output."
            logger.error(error_message)
            return DecisionResult(decision=Decision.RESPOND, message="Manager did not return any result.")

        analysis, data = self.extract_json_from_output(result_text=raw_text)
        if self.manager.streaming.enabled and self.manager.streaming.mode == StreamingMode.ALL:
            self.manager.stream_content(
                content={"analysis": analysis, "data": data},
                step="manager_input_handling",
                source=self.name,
                config=config,
                by_tokens=False,
                **kwargs,
            )

        if not data:
            error_message = f"Orchestrator {self.name} - {self.id}: Failed to extract JSON from manager output."
            logger.error(error_message)
            if attempt >= self.max_user_analyze_retries:
                return DecisionResult(
                    decision=Decision.RESPOND,
                    message="Unable to extract valid JSON from manager output after multiple attempts.",
                )
            _json_prompt_fix = " Please provide a valid JSON response, inside <output>...</output> tags."
            return self._analyze_user_input(input_task + _json_prompt_fix, config=config, attempt=attempt + 1, **kwargs)

        decision_str = data.get("decision", Decision.RESPOND.value)
        message = data.get("message", "")

        try:
            decision = Decision(decision_str)
        except ValueError:
            warning_message = (
                f"Orchestrator {self.name} - {self.id}: Unrecognized decision '{decision_str}', defaulting to RESPOND."
            )
            logger.warning(warning_message)
            decision = Decision.RESPOND

        return DecisionResult(decision=decision, message=message)

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
            input_data={"action": "final", **input_data}, config=config, run_depends=self._run_depends, **kwargs
        )
        self._run_depends = [NodeDependency(node=self.manager).to_dict()]

        if manager_result.status != RunnableStatus.SUCCESS:
            error_message = f"Manager '{self.manager.name}' failed: {manager_result.error.message}"
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

    def parse_xml_content(self, text: str, tag: str) -> str:
        """Extract content from XML-like tags."""
        match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
        return match.group(1).strip() if match else ""

    def _extract_output_content(self, text: str) -> tuple[str, str]:
        """
        Extracts the content of the <analysis> and <output> tags. If a properly closed tag is not found,
        fall back to extracting everything after the first occurrence of <output>.
        """

        output = self.parse_xml_content(text, "output")
        analysis = self.parse_xml_content(text, "analysis")

        if output:
            return analysis, output

        start = text.find("<output>")
        if start != -1:
            fallback_content = text[start + len("<output>") :].strip()
            if fallback_content:
                return analysis, fallback_content
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
