import re
from enum import Enum
from xml.etree import ElementTree as ET  # nosec B405

from pydantic import BaseModel

from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.base import Agent
from dynamiq.nodes.agents.orchestrators.adaptive_manager import AdaptiveAgentManager
from dynamiq.nodes.agents.orchestrators.orchestrator import ActionParseError, Orchestrator, OrchestratorError
from dynamiq.nodes.node import NodeDependency
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.utils.chat import format_chat_history
from dynamiq.utils.logger import logger


class AgentNotFoundError(OrchestratorError):
    """Raised when a specified agent is not found."""

    pass


class ActionCommand(str, Enum):
    DELEGATE = "delegate"
    FINAL_ANSWER = "final_answer"
    RESPOND = "respond"


class Action(BaseModel):
    command: ActionCommand
    agent: str | None = None
    task: str | None = None
    answer: str | None = None


class AdaptiveOrchestrator(Orchestrator):
    """
    Orchestrates the execution of complex tasks using multiple specialized agents.

    This class manages the breakdown of a main objective into subtasks,
    delegates these subtasks to appropriate agents, and synthesizes the results
    into a final answer.

    Attributes:
        manager (ManagerAgent): The managing agent responsible for overseeing the orchestration process.
        agents (List[BaseAgent]): List of specialized agents available for task execution.
        objective (Optional[str]): The main objective of the orchestration.
        max_loops (Optional[int]): Maximum number of actions.
        reflection_enabled (Optional[bool]): Enable reflection mode
    """

    name: str | None = "AdaptiveOrchestrator"
    group: NodeGroup = NodeGroup.AGENTS
    manager: AdaptiveAgentManager
    agents: list[Agent] = []
    max_loops: int = 15
    reflection_enabled: bool = False

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

    def init_components(self, connection_manager: ConnectionManager | None = None) -> None:
        """
        Initialize components of the orchestrator.

        Args:
            connection_manager (Optional[ConnectionManager]): The connection manager. Defaults to ConnectionManager.
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

    def get_next_action(self, config: RunnableConfig = None, **kwargs) -> Action:
        """
        Determine the next action based on the current state and LLM output.

        Returns:
            Action: The next action to be taken.

        Raises:
            ActionParseError: If there is an error parsing the action from the LLM response.
        """

        manager_result = self.manager.run(
            input_data={
                "action": "plan",
                "agents": self.agents_descriptions,
                "chat_history": format_chat_history(self._chat_history),
            },
            config=config,
            run_depends=self._run_depends,
            **kwargs,
        )
        self._run_depends = [NodeDependency(node=self.manager).to_dict()]

        if manager_result.status != RunnableStatus.SUCCESS:
            error_message = f"Agent '{self.manager.name}' failed: {manager_result.output.get('content')}"
            raise ActionParseError(f"Unable to retrieve the next action from Agent Manager, Error: {error_message}")

        manager_content = manager_result.output.get("content").get("result")

        if self.reflection_enabled:
            reflect_result = self.manager.run(
                input_data={
                    "action": "reflect",
                    "agents": self.agents_descriptions,
                    "chat_history": format_chat_history(self._chat_history),
                    "plan": manager_content,
                    "agent_output": "",
                },
                config=config,
                run_depends=self._run_depends,
                **kwargs,
            )
            self._run_depends = [NodeDependency(node=self.manager).to_dict()]

            if reflect_result.status != RunnableStatus.SUCCESS:
                error_message = (
                    f"Agent '{self.manager.name}' failed on reflection: {reflect_result.output.get('content')}"
                )
                logger.error(error_message)
                return self._parse_xml_content(manager_content)
            else:
                reflect_content = reflect_result.output.get("content").get("result")
                try:
                    return self._parse_xml_content(reflect_content)
                except ActionParseError as e:
                    logger.error(f"Agent '{self.manager.name}' failed on reflection parsing: {str(e)}")
                    return self._parse_xml_content(manager_content)
        return self._parse_xml_content(manager_content)

    def run_flow(self, input_task: str, config: RunnableConfig = None, **kwargs) -> str:
        """
        Process the given task using the manager agent logic.

        Args:
            input_task (str): The task to be processed.
            config (RunnableConfig): Configuration for the runnable.

        Returns:
            str: The final answer generated after processing the task.
        """
        self._chat_history.append({"role": "user", "content": input_task})

        for i in range(self.max_loops):
            action = self.get_next_action(config=config, **kwargs)
            logger.info(f"Orchestrator {self.name} - {self.id}: Loop {i + 1} - Action: {action.dict()}")
            if action.command == ActionCommand.DELEGATE:
                self._handle_delegation(action=action, config=config, **kwargs)

            elif action.command == ActionCommand.RESPOND:
                respond_result = self._handle_respond(action=action)
                respond_final_result = self.parse_xml_final_answer(respond_result)
                return respond_final_result

            elif action.command == ActionCommand.FINAL_ANSWER:
                manager_final_result = self.get_final_result(
                    {
                        "input_task": input_task,
                        "chat_history": format_chat_history(self._chat_history),
                        "preliminary_answer": action.answer,
                    },
                    config=config,
                    **kwargs,
                )
                final_result = self.parse_xml_final_answer(manager_final_result)
                return final_result

    def _handle_delegation(self, action: Action, config: RunnableConfig = None, **kwargs) -> None:
        """
        Handle task delegation to a specialized agent.

        Args:
            action (Action): The action containing the delegation command and details.
        """
        agent = next((a for a in self.agents if a.name == action.agent), None)
        if agent:
            result = agent.run(
                input_data={"input": action.task},
                config=config,
                run_depends=self._run_depends,
                **kwargs,
            )
            self._run_depends = [NodeDependency(node=agent).to_dict()]
            if result.status != RunnableStatus.SUCCESS:
                error_message = f"Agent '{agent.name}' failed: {result.output.get('content')}"
                raise OrchestratorError(f"Failed to execute Agent {agent.name}, due to error: {error_message}")

            self._chat_history.append(
                {
                    "role": "system",
                    "content": f"Agent {action.agent} result: {result.output.get('content')}",
                }
            )
        else:
            result = self.manager.run(
                input_data={
                    "action": "respond",
                    "task": action.task,
                    "agents": self.agents_descriptions,
                    "chat_history": format_chat_history(self._chat_history),
                },
                config=config,
                run_depends=self._run_depends,
                **kwargs,
            )
            self._run_depends = [NodeDependency(node=self.manager).to_dict()]
            if result.status != RunnableStatus.SUCCESS:
                logger.error(
                    f"Orchestrator {self.name} - {self.id}: "
                    f"Error executing {self.manager.name}:"
                    f"{result.output.get('content')}"
                )

            self._chat_history.append(
                {
                    "role": "system",
                    "content": f"LLM result: {result.output.get('content')}",
                }
            )

    def _handle_respond(self, action: Action, config: RunnableConfig = None, **kwargs):
        """
        A direct response from the Manager, delegated to the manager agent.
        """

        manager_result = self.manager.run(
            input_data={
                "action": "respond",
                "task": action.task,
                "agents": self.agents_descriptions,
                "chat_history": format_chat_history(self._chat_history),
            },
            config=config,
            run_depends=self._run_depends,
            **kwargs,
        )
        self._run_depends = [NodeDependency(node=self.manager).to_dict()]

        if manager_result.status != RunnableStatus.SUCCESS:
            error_message = f"Manager agent '{self.manager.name}' failed: {manager_result.output.get('content')}"
            raise OrchestratorError(f"Failed to execute respond action with manager, due to error: {error_message}")

        manager_result_content = manager_result.output.get("content").get("result")

        self._chat_history.append(
            {
                "role": "system",
                "content": f"[Manager agent '{self.manager.name} - quick response]: {manager_result_content}",
            }
        )
        return manager_result_content

    def setup_streaming(self) -> None:
        """Setups streaming for orchestrator."""
        self.manager.streaming = self.streaming
        for agent in self.agents:
            agent.streaming = self.streaming

    def _parse_xml_content(self, content: str) -> Action:
        """
        Parse XML content to extract action details.

        Args:
            content (str): XML formatted content from LLM response

        Returns:
            Action: Parsed action object

        Raises:
            ActionParseError: If XML parsing fails or required tags are missing
        """
        try:
            content = content.replace("```", "").strip()

            output_match = re.search(r"<output>(.*?)</output>", content, re.DOTALL)
            if not output_match:
                error_message = (
                    f"Orchestrator {self.name} - {self.id}: XML parsing error: No <output> tags found in the response"
                )
                raise ActionParseError(error_message)

            output_content = output_match.group(1).strip()
            root = ET.fromstring(f"<root>{output_content}</root>")  # nosec B314

            action_elem = root.find("action")
            if action_elem is None:
                error_message = (
                    f"Orchestrator {self.name} - {self.id}: XML parsing error: No <action> tag found in the response"
                )
                raise ActionParseError(error_message)

            action_type = action_elem.text.strip().lower()

            if action_type == "delegate":
                agent_elem = root.find("agent")
                task_elem = root.find("task")
                task_data_elem = root.find("task_data")

                if agent_elem is None or task_elem is None:
                    error_message = (
                        f"Orchestrator {self.name} - {self.id}: XML parsing error: "
                        f"Delegate action missing required <agent> or <task> tags"
                    )
                    raise ActionParseError(error_message)

                return Action(
                    command=ActionCommand.DELEGATE,
                    agent=agent_elem.text.strip(),
                    task=task_elem.text.strip()
                    + (f" {task_data_elem.text.strip()}" if task_data_elem is not None else ""),
                )

            elif action_type == "final_answer":
                answer_elem = root.find("final_answer")
                if answer_elem is None:
                    error_message = (
                        f"Orchestrator {self.name} - {self.id}: XML parsing error: "
                        f"Final answer action missing <final_answer> tag"
                    )
                    raise ActionParseError(error_message)

            elif action_type == "respond":
                task_elem = root.find("task")

                if task_elem is None:
                    error_message = (
                        f"Orchestrator {self.name} - {self.id}: XML parsing error: Respond action missing <task> tag"
                    )
                    raise ActionParseError(error_message)

                return Action(
                    command=ActionCommand.RESPOND,
                    task=task_elem.text.strip(),
                )
            else:
                raise ActionParseError(f"Unknown action type: {action_type}")

            return Action(command=ActionCommand.FINAL_ANSWER, answer=answer_elem.text.strip())

        except ET.ParseError as e:
            error_message = f"Orchestrator {self.name} - {self.id}: XML parsing error: {str(e)}"
            raise ActionParseError(error_message)
        except Exception as e:
            error_message = f"Orchestrator {self.name} - {self.id}: Error parsing action: {str(e)}"
            raise ActionParseError(error_message)

    def parse_xml_final_answer(self, content: str) -> str:
        output_match = re.search(r"<output>(.*?)</output>", content, re.DOTALL)
        if not output_match:
            error_message = (
                f"Orchestrator {self.name} - {self.id}: Error parsing final answer: "
                f"No <output> tags found in the response {content}"
            )
            raise ActionParseError(error_message)

        output_content = output_match.group(1).strip()
        return output_content
