import re
from functools import cached_property
from typing import Any

from pydantic import BaseModel, TypeAdapter

from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.agents.base import Agent
from dynamiq.nodes.agents.orchestrators.linear_manager import LinearAgentManager
from dynamiq.nodes.node import NodeDependency, ensure_config
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.utils.logger import logger


class Task(BaseModel):
    """
    Represents a single task in the LinearOrchestrator system.

    Attributes:
        id (int): Unique identifier for the task.
        name (str): Name of the task.
        description (str): Detailed description of the task.
        dependencies (list[int]): List of task IDs that this task depends on.
        output: dict[str, Any] | str: Expected output of the task.
    """

    id: int
    name: str
    description: str
    dependencies: list[int]
    output: dict[str, Any] | str


class LinearOrchestrator(Node):
    """
    Manages the execution of tasks by coordinating multiple agents and leveraging LLM (Large Language Model).

    Attributes:
        manager (ManagerAgent): The managing agent responsible for overseeing the orchestration process.
        agents (List[BaseAgent]): List of specialized agents available for task execution.
        input_task (str | None): Initial task input.
        use_summarizer (bool): Indicates if a final summarizer is used.
        summarize_all_answers (bool): Indicates whether to summarize answers to all tasks or use only last one.\
              Will only be applied if use_summarizer is set to True.

    """

    name: str | None = "LinearOrchestrator"
    group: NodeGroup = NodeGroup.AGENTS
    manager: LinearAgentManager
    agents: list[Agent] = []
    input_task: str | None = None
    use_summarizer: bool = True
    summarize_all_answers: bool = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._results = {}
        self._run_depends = []

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

    def init_components(
        self, connection_manager: ConnectionManager = ConnectionManager()
    ):
        """Initialize components for the manager and agents."""
        super().init_components(connection_manager)
        if self.manager.is_postponed_component_init:
            self.manager.init_components(connection_manager)

        for agent in self.agents:
            if agent.is_postponed_component_init:
                agent.init_components(connection_manager)

    @cached_property
    def agents_descriptions(self) -> str:
        """Generate a string description of all agents."""
        return (
            "\n".join([f"{i}. {_agent.name}" for i, _agent in enumerate(self.agents)])
            if self.agents
            else ""
        )

    def get_tasks(self, config: RunnableConfig = None, **kwargs) -> list[Task]:
        """Generate tasks using the manager agent."""
        manager_result = self.manager.run(
            input_data={
                "action": "plan",
                "input_task": self.input_task,
                "agents": self.agents_descriptions,
            },
            config=config,
            run_depends=self._run_depends,
            **kwargs,
        )
        self._run_depends = [NodeDependency(node=self.manager).to_dict()]

        if manager_result.status != RunnableStatus.SUCCESS:
            raise ValueError("Agent LLM failed to generate tasks")

        manager_result_content = manager_result.output.get("content").get("result")
        logger.debug(
            f"LinearOrchestrator {self.id}: Manager plan result content: {manager_result_content}"
        )

        tasks = self.parse_tasks_from_output(manager_result_content)
        logger.debug(f"LinearOrchestrator {self.id}: Task list JSON: {tasks}")

        return tasks

    def parse_tasks_from_output(self, output: str) -> list[Task]:
        """Parse tasks from the manager's output string."""
        # Remove 'output' XML tags if present
        if "<output>" in output and "</output>" in output:
            output = output.split("<output>")[1].split("</output>")[0]

        # Remove '```' code block markers and 'json' keyword if present
        try:
            output = output.replace("```", "").replace("json", "")
        except AttributeError as e:
            logger.warning(
                f"LinearOrchestrator {self.id}: Failed to remove code block markers and 'json' keyword from output {e}"
            )

        # Parse the JSON string
        try:
            task_list_json = output.strip()
        except AttributeError as e:
            logger.warning(
                f"LinearOrchestrator {self.id}: Failed to strip the output string: {e}"
            )
            task_list_json = output
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
                dependencies_formatted += (
                    f"**Task:** {task_name}\n**Result:** {task_result}\n\n"
                )

        return dependencies_formatted.strip()

    def get_final_result(self, config: RunnableConfig = None, **kwargs) -> str:
        """Generate the final result."""
        final_task_output = "\n\n".join(
            f"**Task:** {result['name']}\n**Result:** {result['result']}"
            for result in self._results.values()
            if result
        )

        if self.use_summarizer:
            if not self.summarize_all_answers:
                final_task_id = max(self._results.keys(), default=None)
                logger.debug(f"LinearOrchestrator {self.id}: Final task id: {final_task_id}")

                if final_task_id is not None:
                    final_task_output = self._results[final_task_id].get("result", "")

                logger.debug(f"LinearOrchestrator {self.id}: Final task output: {final_task_output}")

            logger.debug(f"LinearOrchestrator {self.id}: Running final summarizer")
            manager_result = self.manager.run(
                input_data={
                    "action": "final",
                    "input_task": self.input_task,
                    "tasks_outputs": final_task_output,
                },
                config=config,
                run_depends=self._run_depends,
                **kwargs,
            )
            self._run_depends = [NodeDependency(node=self.manager).to_dict()]
            if manager_result.status != RunnableStatus.SUCCESS:
                raise ValueError("Agent LLM failed to generate final result")

            result = manager_result.output.get("content")
        else:
            logger.debug(f"LinearOrchestrator {self.id}: Returning final task output")
            result = final_task_output

        return result

    def run_tasks(
        self, tasks: list[Task], config: RunnableConfig = None, **kwargs
    ) -> None:
        """Execute the tasks using appropriate agents."""
        logger.debug(
            f"LinearOrchestrator {self.id}: Assigning and executing tasks. Agents: {self.agents_descriptions}"
        )

        for task in tasks:
            task_per_llm = f"**{task.description}**\n**Required information for output**: {task.output}"
            logger.debug(
                f"LinearOrchestrator {self.id}: task {task.id}.{task.name} prepared for LLM: {task_per_llm}"
            )

            dependency_outputs = self.get_dependency_outputs(task.dependencies)
            if dependency_outputs:
                task_per_llm += f"\n{dependency_outputs}"
                logger.debug(
                    f"LinearOrchestrator {self.id}: task {task.id}.{task.name} "
                    f"prepared for LLM with dependencies: {task_per_llm}"
                )

            logger.debug(
                f"LinearOrchestrator {self.id}: task {task.id}.{task.name} "
                f"with dependencies: {task.dependencies}"
            )
            logger.debug(
                f"LinearOrchestrator {self.id}: task {task.id}.{task.name} "
                f"task dependencies output: ```{dependency_outputs}```"
            )

            success_flag = False
            for _ in range(self.manager.max_loops):
                manager_result = self.manager.run(
                    input_data={
                        "action": "assign",
                        "input_task": self.input_task,
                        "task": task_per_llm,
                        "agents": self.agents_descriptions,
                    },
                    config=config,
                    run_depends=self._run_depends,
                    **kwargs,
                )
                self._run_depends = [NodeDependency(node=self.manager).to_dict()]

                logger.debug(f"LinearOrchestrator {self.id}: Assigner LLM result: {manager_result}")

                if manager_result.status == RunnableStatus.SUCCESS:
                    try:
                        assigned_agent_index = int(manager_result.output.get("content").get("result", -1))

                    except ValueError:
                        logger.warning(
                            f"LinearOrchestrator {self.id}: Invalid agent index: {manager_result.output.get('content').get('result', -1)}"  # noqa: E501
                        )
                        try:
                            match = re.match(
                                r"^\d+",
                                manager_result.output.get("content").get("result", -1),
                            )
                            assigned_agent_index = int(match.group())
                        except Exception as e:
                            logger.error(f"LinearOrchestrator {self.id}: Failed to extract agent index: {e}")
                            assigned_agent_index = -1

                    if 0 <= assigned_agent_index < len(self.agents):
                        assigned_agent = self.agents[assigned_agent_index]
                        logger.debug(
                            f"LinearOrchestrator {self.id}: Execute task {task.id}.{task.name} "
                            f"by agent {assigned_agent_index}.{assigned_agent.name}"
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
                            )

                        self._results[task.id] = {
                            "name": task.name,
                            "result": result.output["content"],
                        }
                        logger.debug(
                            f"LinearOrchestrator {self.id}: Task {task.id}.{task.name} "
                            f"executed by agent {assigned_agent_index}"
                        )
                        logger.debug(
                            f"LinearOrchestrator {self.id}: Task {task.id}.{task.name}\
                                output: {result.output['content']}"
                        )
                        success_flag = True
                        break
                task_per_llm += f"Error occured {manager_result.output}"

            if success_flag:
                continue

            else:
                raise ValueError(f"Failed to assign task {task.id}.{task.name} by Manager Agent")

    def execute(self, input_data: Any, config: RunnableConfig = None, **kwargs) -> dict:
        """
        Execute the LinearOrchestrator flow.

        Args:
            input_data (Any): The input data containing the objective or additional context.
            config (Optional[RunnableConfig]): Configuration for the runnable.
            **kwargs: Additional keyword arguments.

        Returns:
            dict[str, Any]: The result of the orchestration process.
        """
        self.reset_run_state()
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        self.input_task = input_data.get("input") or self.input_task

        logger.debug(
            f"LinearOrchestrator {self.id}: starting the flow with input_task:\n```{self.input_task}```"
        )
        run_kwargs = kwargs | {"parent_run_id": kwargs.get("run_id")}
        run_kwargs.pop("run_depends", None)
        if self.streaming.enabled:
            self.manager.streaming = self.streaming
            for agent in self.agents:
                agent.streaming = self.streaming

        tasks = self.get_tasks(config=config, **run_kwargs)
        logger.debug(f"LinearOrchestrator {self.id}: tasks initialized:\n '{tasks}'")
        self.run_tasks(tasks=tasks, config=config, **run_kwargs)
        logger.debug(f"LinearOrchestrator {self.id}: tasks assigned and executed.")
        result = self.get_final_result(config=config, **run_kwargs)

        logger.debug(f"LinearOrchestrator {self.id}: output collected")
        return {"content": result}
