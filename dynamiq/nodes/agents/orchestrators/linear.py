import json
import re
from enum import Enum
from functools import cached_property
from typing import Any

from pydantic import BaseModel, Field, TypeAdapter

from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.base import Agent
from dynamiq.nodes.agents.orchestrators.linear_manager import LinearAgentManager
from dynamiq.nodes.agents.orchestrators.orchestrator import ActionParseError, Orchestrator, OrchestratorError
from dynamiq.nodes.node import NodeDependency
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types.feedback import PlanApprovalConfig
from dynamiq.utils.logger import logger


class Task(BaseModel):
    """
    Represents a single task in the LinearOrchestrator system.

    Attributes:
        id (int): Unique identifier for the task
        name (str): Name of the task
        description (str): Detailed description of the task
        dependencies (list[int]): List of task IDs that this task depends on
        output (Union[dict[str, Any], str]): Expected output of the task,
            either as a structured dictionary or string
    """

    id: int
    name: str
    description: str
    dependencies: list[int]
    output: dict[str, Any] | str


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


class LinearOrchestrator(Orchestrator):
    """
    Manages the execution of tasks by coordinating multiple agents and leveraging LLM (Large Language Model).

    Attributes:
        name (str | None): Name of the orchestrator.
        group (NodeGroup): The group this node belongs to.
        manager (LinearAgentManager): The managing agent responsible for overseeing the orchestration process.
        agents (list[Agent]): List of specialized agents available for task execution.
        use_summarizer (bool): Indicates if a final summarizer is used.
        summarize_all_answers (bool): Indicates whether to summarize answers to all tasks
            or use only last one. Will only be applied if use_summarizer is set to True.
        max_plan_retries (int): Maximum number of plan generation retries.
        plan_approval (PlanApprovalConfig): Configuration for plan approval.
        max_user_analyze_retries (int): Maximum number of retries for analyzing user input.
    """

    name: str | None = "LinearOrchestrator"
    group: NodeGroup = NodeGroup.AGENTS
    manager: LinearAgentManager
    agents: list[Agent] = []
    use_summarizer: bool = True
    summarize_all_answers: bool = False
    max_plan_retries: int = 5
    plan_approval: PlanApprovalConfig = Field(default_factory=PlanApprovalConfig)
    max_user_analyze_retries: int = 3

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._results = {}

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
        self._results = {}
        self._run_depends = []
        self._chat_history = []

    def init_components(self, connection_manager: ConnectionManager | None = None):
        """
        Initialize components for the manager and agents.

        Args:
            connection_manager (Optional[ConnectionManager]): The connection manager. Defaults to ConnectionManager.
        """
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        if self.manager.is_postponed_component_init:
            self.manager.init_components(connection_manager)

        for agent in self.agents:
            if agent.is_postponed_component_init:
                agent.init_components(connection_manager)

    @cached_property
    def agents_descriptions(self) -> str:
        """Generate a string description of all agents."""
        return "\n".join([f"{i}. {_agent.name}" for i, _agent in enumerate(self.agents)]) if self.agents else ""

    def get_tasks(self, input_task: str, config: RunnableConfig = None, **kwargs) -> list[Task]:
        """
        Generate tasks using the manager agent.

        Args:
            input_task (str): The input task to generate subtasks from
            config (RunnableConfig, optional): Configuration for the runnable
            **kwargs: Additional keyword arguments passed to the manager's run method

        Returns:
            list[Task]: List of generated tasks

        Raises:
            ValueError: If task generation fails
            OrchestratorError: If maximum number of retries is reached
        """
        manager_result_content = ""
        feedback = ""

        for _ in range(self.max_plan_retries):
            manager_result = self.manager.run(
                input_data={
                    "action": "plan",
                    "input_task": input_task,
                    "agents": self.agents_descriptions,
                    "feedback": feedback,
                    "previous_plan": manager_result_content,
                },
                config=config,
                run_depends=self._run_depends,
                **kwargs,
            )
            self._run_depends = [NodeDependency(node=self.manager).to_dict()]

            if manager_result.status != RunnableStatus.SUCCESS:
                error_message = f"LLM '{self.manager.name}' failed: {manager_result.output.get('content')}"
                raise ValueError(f"Failed to generate tasks: {error_message}")

            manager_result_content = manager_result.output.get("content").get("result")
            logger.info(
                f"Orchestrator {self.name} - {self.id}: Tasks generated by {self.manager.name} - {self.manager.id}:"
                f"\n{manager_result_content}"
            )
            try:
                tasks = self.parse_tasks_from_output(manager_result_content)

            except ActionParseError as e:
                feedback = str(e)
                continue

            if not self.plan_approval.enabled:
                return tasks
            else:
                approval_result = self.send_approval_message(
                    self.plan_approval, {"tasks": tasks}, config=config, **kwargs
                )

                feedback = approval_result.feedback
                if approval_result.is_approved:
                    return approval_result.data.get("tasks")

        raise OrchestratorError("Maximum number of loops reached for generating plan.")

    def parse_tasks_from_output(self, output: str) -> list[Task]:
        """Parse tasks from the manager's output string."""

        output_match = re.search(r"<output>(.*?)</output>", output, re.DOTALL)
        if not output_match:
            error_response = f"Error parsing final answer: No <output> tags found in the response {output}"
            raise ActionParseError(f"Error: {error_response}")

        output_content = output_match.group(1).strip()

        try:
            output_content = self._clean_output(output_content)
        except AttributeError as e:
            logger.warning(
                f"Orchestrator {self.name} - {self.id}: "
                f"Failed to remove code block markers and 'json' keyword "
                f"from output {output_content} due to error: {e}"
            )

        try:
            task_list_json = output_content.strip()
        except AttributeError as e:
            logger.warning(
                f"Orchestrator {self.name} - {self.id}: Failed to strip the output {output_content} due to error: {e}"
            )
            task_list_json = output_content
        return TypeAdapter(list[Task]).validate_json(task_list_json)

    def get_dependency_outputs(self, dependencies: list[int]) -> str:
        """Format the outputs of dependent tasks."""
        if not dependencies:
            return ""

        dependencies_formatted = "**Here is the previously collected information:**\n"
        for dep in dependencies:
            if dep in self._results:
                task_name = self._results[dep]["name"]
                task_result = str(self._results[dep]["result"])
                dependencies_formatted += f"**Task:** {task_name}\n**Result:** {task_result}\n\n"

        return dependencies_formatted.strip()

    def run_tasks(self, tasks: list[Task], input_task: str, config: RunnableConfig = None, **kwargs) -> None:
        """Execute the tasks using appropriate agents."""

        for count, task in enumerate(tasks, start=1):
            task_per_llm = f"**{task.description}**\n**Required information for output**: {task.output}"

            dependency_outputs = self.get_dependency_outputs(task.dependencies)
            if dependency_outputs:
                task_per_llm += f"\n{dependency_outputs}"

            success_flag = False
            for _ in range(self.manager.max_loops):
                manager_result = self.manager.run(
                    input_data={
                        "action": "assign",
                        "input_task": input_task,
                        "task": task_per_llm,
                        "agents": self.agents_descriptions,
                    },
                    config=config,
                    run_depends=self._run_depends,
                    **kwargs,
                )
                self._run_depends = [NodeDependency(node=self.manager).to_dict()]

                if manager_result.status == RunnableStatus.SUCCESS:
                    assigned_agent_index = self._extract_agent_index(manager_result.output.get("content", {}))

                    if 0 <= assigned_agent_index < len(self.agents):
                        assigned_agent = self.agents[assigned_agent_index]
                        logger.info(
                            f"Orchestrator {self.name} - {self.id}: Loop {count} - "
                            f"Assigned agent: {assigned_agent.name} - {assigned_agent.id}"
                        )
                        result = assigned_agent.run(
                            input_data={"input": task_per_llm},
                            config=config,
                            run_depends=self._run_depends,
                            **kwargs,
                        )
                        self._run_depends = [NodeDependency(node=assigned_agent).to_dict()]
                        if result.status != RunnableStatus.SUCCESS:
                            raise ValueError(
                                f"Failed to execute task {task.id}.{task.name} "
                                f"by agent {assigned_agent_index}.{assigned_agent.name}"
                                f"due to error: {result.output.get('content')}"
                            )

                        self._results[task.id] = {
                            "name": task.name,
                            "result": result.output["content"],
                        }

                        success_flag = True
                        break
                task_per_llm += f"Error occurred:{manager_result.output}"

            if success_flag:
                continue

            else:
                raise ValueError(
                    f"Orchestrator {self.name} - {self.id}: "
                    f"Failed to assign task {task.id}.{task.name} "
                    f"by Manager Agent due to error: {manager_result.output}"
                )

    def generate_final_answer(self, task: str, config: RunnableConfig, **kwargs) -> str:
        """
        Generates final answer using the manager agent logic.

        Args:
            task (str): The task to be processed.
            config (RunnableConfig): Configuration for the runnable.

        Returns:
            str: The final answer generated after processing the task.
        """
        tasks_outputs = "\n\n".join(
            f"**Task:** {result['name']}\n**Result:** {result['result']}" for result in self._results.values() if result
        )

        if self.use_summarizer:
            if not self.summarize_all_answers:
                final_task_id = max(self._results.keys(), default=None)

                if final_task_id is not None:
                    final_task_output = self._results[final_task_id].get("result", "")
                    logger.debug(f"Orchestrator {self.name} - {self.id}: Final task output: {final_task_output}")
                    return final_task_output

            final_result_content = self.get_final_result(
                {"input_task": task, "chat_history": self._chat_history, "tasks_outputs": tasks_outputs},
                config=config,
                **kwargs,
            )

            try:
                final_result = re.search(r"<final_answer>(.*?)</final_answer>", final_result_content, re.DOTALL)
                final_result_answer = final_result.group(1).strip()
                return final_result_answer
            except Exception as e:
                error_response = f"Orchestrator {self.name} - {self.id}: Error parsing final answer: {e}"
                logger.error(error_response)
                if "final_answer" in final_result_content:
                    logger.info(f"Orchestrator {self.name} - {self.id}: Return raw answer")
                    return final_result_content
                else:
                    raise ActionParseError(f"{error_response}")
        return tasks_outputs

    def run_flow(self, input_task: str, config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        """
        Process the given task using the manager agent logic.

        Args:
            input_task (str): The task to be processed.
            config (RunnableConfig): Configuration for the runnable.

        Returns:
            dict[str, Any]: The final output generated after processing the task.
        """
        analysis = self._analyze_user_input(input_task, config=config, **kwargs)
        decision = analysis.decision
        message = analysis.message

        if decision == Decision.RESPOND:
            return {"content": message}
        else:
            tasks = self.get_tasks(input_task, config=config, **kwargs)
            self.run_tasks(tasks=tasks, input_task=input_task, config=config, **kwargs)
            return {"content": self.generate_final_answer(input_task, config, **kwargs)}

    def setup_streaming(self) -> None:
        """Setups streaming for orchestrator."""
        self.manager.streaming = self.streaming
        for agent in self.agents:
            agent.streaming = self.streaming

    def _analyze_user_input(self, input_task: str, config: RunnableConfig = None, attempt: int = 1,
                            **kwargs) -> DecisionResult:
        """
        Calls the manager's 'handle_input' action to decide if we should respond
        immediately or proceed with a plan.

        Args:
            input_task (str): The user's input or task description.
            config: Optional configuration object passed to the manager's run method.
            attempt (int): The current attempt number for analyzing user input.
            **kwargs: Additional keyword arguments passed to the manager's run method.

        Returns:
            DecisionResult: An object containing the decision (as an Enum) and a message.
        """

        handle_result = self.manager.run(
            input_data={
                "action": "handle_input",
                "task": input_task,
                "agents": self.agents_descriptions,
            },
            config=config,
            run_depends=[],
            **kwargs,
        )

        if handle_result.status != RunnableStatus.SUCCESS:
            error_message = (
                f"Orchestrator {self.name} - {self.id}: Manager failed to analyze input: {handle_result.output}"
            )
            logger.error(error_message)
            return DecisionResult(decision=Decision.RESPOND, message=f"Error analyzing request: {handle_result.output}")

        content = handle_result.output.get("content", {})
        raw_text = content.get("result", "")
        if not raw_text:
            error_message = f"Orchestrator {self.name} - {self.id}: No 'result' field in manager output."
            logger.error(error_message)
            return DecisionResult(decision=Decision.RESPOND, message="Manager did not return any result.")

        data = self.extract_json_from_output(result_text=raw_text)
        if not data:
            error_message = f"Orchestrator {self.name} - {self.id}: Failed to extract JSON from manager output."
            logger.error(error_message)
            if attempt >= self.max_user_analyze_retries:
                return DecisionResult(
                    decision=Decision.RESPOND,
                    message="Unable to extract valid JSON from manager output after multiple attempts."
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

    def extract_json_from_output(self, result_text: str) -> dict | None:
        """
        Extracts JSON data from the given text by looking for content within
        <output>...</output> tags. Strips any Markdown code block fences.

        Args:
            result_text (str): The text from which to extract JSON data.

        Returns:
            dict | None: The extracted JSON dictionary if successful, otherwise None.
        """
        output_content = self._extract_output_content(result_text)
        output_content = self._clean_output(output_content)

        try:
            data = json.loads(output_content)
            return data
        except json.JSONDecodeError as e:
            error_message = f"Orchestrator {self.name} - {self.id}: JSON decoding error: {e}"
            logger.error(error_message)
            return None

    def _clean_output(self, text: str) -> str:
        """Remove Markdown code fences and extra whitespace from a text."""
        cleaned = re.sub(r"^```(?:json)?\s*|```$", "", text).strip()
        return cleaned

    def _extract_agent_index(self, result_content: dict[str, Any]) -> int:
        """
        Extracts and validates the agent index from the result content.

        Args:
            result_content (dict[str, Any]): The content containing the agent index

        Returns:
            int: The extracted agent index, or -1 if extraction fails
        """
        raw = result_content.get("result", -1)
        try:
            return int(raw)
        except ValueError:
            logger.warning(f"Invalid agent index: {raw}")
            match = re.match(r"^\d+", str(raw))
            if match:
                return int(match.group())
            else:
                logger.error(f"Failed to extract agent index from: {raw}")
                return -1
