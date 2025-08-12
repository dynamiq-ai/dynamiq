import asyncio
import json
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.agents.base import Agent
from dynamiq.nodes.agents.orchestrators.concurrent_manager import ConcurrentAgentManager
from dynamiq.nodes.agents.orchestrators.orchestrator import Decision, Orchestrator, OrchestratorError
from dynamiq.nodes.node import NodeDependency
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.utils.chat import format_chat_history
from dynamiq.utils.logger import logger


class TaskComplexity(str, Enum):
    """Enumeration for task complexity levels."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class TaskType(str, Enum):
    """Enumeration for task execution types."""

    AGENT = "agent"
    TOOL = "tool"
    HYBRID = "hybrid"


class ConversationDecision(str, Enum):
    """Enumeration for conversation-level decisions."""

    RESPOND = "respond"
    PLAN = "plan"
    CLARIFY = "clarify"


class ConversationAnalysis(BaseModel):
    """Represents the analysis of a conversational input."""

    decision: ConversationDecision
    reasoning: str
    message: str
    suggested_next_steps: list[str] = Field(default_factory=list)
    confidence: str = "medium"


class ParallelTask(BaseModel):
    """Represents a task that can be executed in parallel."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str
    complexity: TaskComplexity
    task_type: TaskType
    agent_name: str | None = None
    tool_name: str | None = None
    dependencies: list[str] = Field(default_factory=list)
    context_requirements: list[str] = Field(default_factory=list)
    priority: int = 1
    estimated_duration: int = 60


class TaskResult(BaseModel):
    """Represents the result of a completed task."""

    task_id: str
    success: bool
    result: Any = None
    error: str | None = None
    execution_time: float
    context_updates: dict[str, Any] = Field(default_factory=dict)


class ParallelExecutionPlan(BaseModel):
    """Represents a plan for parallel execution."""

    tasks: list[ParallelTask]
    execution_groups: list[list[str]]
    shared_context: dict[str, Any] = Field(default_factory=dict)
    max_concurrency: int = 3
    timeout: int = 300


class ConcurrentOrchestrator(Orchestrator):
    """
    Advanced orchestrator that combines natural conversation with intelligent parallel task execution.

    This orchestrator starts with conversational interaction and adaptively scales up to complex
    parallel execution when needed. It provides the best of both worlds: natural chat for simple
    interactions and sophisticated orchestration for complex tasks.

    Key Features:
    - Natural conversation handling with adaptive complexity detection
    - Seamless escalation from chat to parallel orchestration
    - Async parallel execution of independent tasks
    - Task complexity analysis for optimal resource allocation
    - Context sharing across conversations and tasks
    - Direct tool integration alongside agent delegation
    - Progressive task building and refinement
    - Streaming support for real-time updates
    - Comprehensive error handling and recovery
    """

    name: str | None = "ConcurrentOrchestrator"
    group: NodeGroup = NodeGroup.AGENTS
    manager: ConcurrentAgentManager
    agents: list[Agent] = []
    tools: list[Node] = []
    max_loops: int = 10
    max_concurrency: int = 3
    task_timeout: int = 300
    enable_complexity_analysis: bool = True
    enable_context_sharing: bool = True
    conversation_mode: bool = True
    orchestration_threshold: float = 0.7
    enable_adaptive_planning: bool = True
    plan_revision_threshold: int = 3
    max_plan_revisions: int = 3

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._execution_results: dict[str, TaskResult] = {}
        self._shared_context: dict[str, Any] = {}
        self._active_tasks: dict[str, asyncio.Task] = {}
        self._conversation_context: dict[str, Any] = {}
        self._current_plan: ParallelExecutionPlan | None = None
        self._plan_revisions: int = 0
        self._completed_tasks_since_revision: int = 0
        self._original_task: str = ""

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {"manager": True, "agents": True, "tools": True}

    def to_dict(self, **kwargs) -> dict:
        """Converts the instance to a dictionary."""
        data = super().to_dict(**kwargs)
        data["manager"] = self.manager.to_dict(**kwargs)
        data["agents"] = [agent.to_dict(**kwargs) for agent in self.agents]
        data["tools"] = [tool.to_dict(**kwargs) for tool in self.tools]
        return data

    def reset_run_state(self):
        """Reset the orchestrator's run state."""
        super().reset_run_state()
        self._execution_results.clear()
        self._shared_context.clear()
        self._active_tasks.clear()
        self._current_plan = None
        self._plan_revisions = 0
        self._completed_tasks_since_revision = 0
        self._original_task = ""

    def init_components(self, connection_manager: ConnectionManager | None = None) -> None:
        """Initialize components of the orchestrator."""
        super().init_components(connection_manager)

        if self.manager.is_postponed_component_init:
            self.manager.init_components(connection_manager)

        for agent in self.agents:
            if agent.is_postponed_component_init:
                agent.init_components(connection_manager)

        for tool in self.tools:
            if tool.is_postponed_component_init:
                tool.init_components(connection_manager)

    @property
    def agents_descriptions(self) -> str:
        """Get a formatted string of agent descriptions."""
        return "\n".join([f"{i}. {agent.name}" for i, agent in enumerate(self.agents)]) if self.agents else ""

    @property
    def tools_descriptions(self) -> str:
        """Get a formatted string of tool descriptions."""
        return "\n".join([f"{i}. {tool.name}" for i, tool in enumerate(self.tools)]) if self.tools else ""

    def analyze_conversation(self, user_input: str, config: RunnableConfig = None, **kwargs) -> ConversationAnalysis:
        """
        Analyze the conversational input to determine the appropriate response approach.

        Args:
            user_input: The user's message or request
            config: Configuration for the runnable
            **kwargs: Additional keyword arguments

        Returns:
            ConversationAnalysis: Analysis of how to handle the input
        """
        if not self.conversation_mode:

            return ConversationAnalysis(
                decision=ConversationDecision.PLAN,
                reasoning="Conversation mode disabled, proceeding with orchestration",
                message="",
                confidence="high",
            )

        logger.info(f"Orchestrator {self.name} - {self.id}: Analyzing conversation input: {user_input[:100]}...")

        try:
            manager_result = self.manager.run(
                input_data={
                    "action": "analyze",
                    "user_input": user_input,
                    "agents": self.agents_descriptions,
                    "tools": self.tools_descriptions,
                    "chat_history": format_chat_history(self._chat_history),
                },
                config=config,
                run_depends=self._run_depends,
                **kwargs,
            )
            self._run_depends = [NodeDependency(node=self.manager).to_dict()]

            if manager_result.status != RunnableStatus.SUCCESS:
                logger.warning(f"Manager analysis failed: {manager_result.error.message}")
                return self._fallback_conversation_analysis(user_input)

            analysis_data_raw = manager_result.output.get("content")
            if isinstance(analysis_data_raw, dict) and "result" in analysis_data_raw:
                analysis_content = analysis_data_raw["result"]
            else:
                analysis_content = analysis_data_raw
            logger.debug(f"Raw conversation analysis content: {analysis_content}")
            analysis_data = self._parse_conversation_analysis(analysis_content)
            return ConversationAnalysis(**analysis_data)

        except Exception as e:
            logger.error(f"Failed to analyze conversation: {e}")
            return self._fallback_conversation_analysis(user_input)

    def _fallback_conversation_analysis(self, user_input: str) -> ConversationAnalysis:
        """Provide fallback conversation analysis when the manager fails."""

        user_lower = user_input.lower().strip()

        if any(greeting in user_lower for greeting in ["hello", "hi", "hey", "good morning", "good afternoon"]):
            return ConversationAnalysis(
                decision=ConversationDecision.RESPOND,
                reasoning="Simple greeting detected",
                message="Hello! How can I help you today?",
                confidence="high",
            )

        if len(user_input.split()) < 3:
            return ConversationAnalysis(
                decision=ConversationDecision.CLARIFY,
                reasoning="Input too brief, needs clarification",
                message="Could you tell me more about what you'd like help with?",
                confidence="medium",
            )

        if any(indicator in user_lower for indicator in ["and", "also", "then", "analyze", "research", "create"]):
            return ConversationAnalysis(
                decision=ConversationDecision.PLAN,
                reasoning="Complex multi-part request detected",
                message="",
                confidence="medium",
            )

        return ConversationAnalysis(
            decision=ConversationDecision.RESPOND,
            reasoning="Default conversational response",
            message="I'd be happy to help! What would you like to work on?",
            confidence="low",
        )

    def _parse_conversation_analysis(self, analysis_content: str) -> dict[str, Any]:
        """Parse the conversation analysis from manager output."""
        logger.debug(f"Parsing conversation analysis from: {analysis_content[:200]}...")

        json_content = None

        if "```json" in analysis_content:
            start = analysis_content.find("```json") + len("```json")
            end = analysis_content.find("```", start)
            json_content = analysis_content[start:end].strip()
        elif "<output>" in analysis_content and "</output>" in analysis_content:
            start = analysis_content.find("<output>") + len("<output>")
            end = analysis_content.find("</output>")
            content = analysis_content[start:end].strip()

            content = content.replace("```json", "").replace("```", "").strip()
            json_content = content
        else:

            start = analysis_content.find("{")
            end = analysis_content.rfind("}") + 1
            if start != -1 and end > start:
                json_content = analysis_content[start:end]
            else:
                logger.error(f"No JSON content found in analysis: {analysis_content}")
                raise ValueError("No JSON content found in analysis")

        logger.debug(f"Extracted JSON content: {json_content}")

        try:
            return json.loads(json_content)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error: {e}, trying to fix common issues")
            logger.warning(f"Problematic JSON: {json_content}")

            fixed_content = json_content.replace("'", '"').replace("True", "true").replace("False", "false")
            try:
                return json.loads(fixed_content)
            except json.JSONDecodeError as e2:
                logger.error(f"Still failed to parse JSON after fixes: {e2}")
                logger.error(f"Final JSON content: {fixed_content}")
                raise

    def analyze_task_complexity(self, task_description: str) -> TaskComplexity:
        """
        Analyze the complexity of a task based on various factors.

        Args:
            task_description: Description of the task to analyze

        Returns:
            TaskComplexity: The determined complexity level
        """
        if not self.enable_complexity_analysis:
            return TaskComplexity.MODERATE

        description_lower = task_description.lower()

        simple_indicators = [
            "get",
            "fetch",
            "retrieve",
            "find",
            "search",
            "lookup",
            "check",
            "calculate",
            "convert",
            "format",
            "validate",
            "verify",
        ]

        complex_indicators = [
            "analyze",
            "evaluate",
            "compare",
            "synthesize",
            "generate",
            "create",
            "design",
            "plan",
            "optimize",
            "integrate",
            "coordinate",
            "orchestrate",
        ]

        simple_count = sum(1 for indicator in simple_indicators if indicator in description_lower)
        complex_count = sum(1 for indicator in complex_indicators if indicator in description_lower)

        word_count = len(task_description.split())

        if word_count > 100 or complex_count > simple_count:
            return TaskComplexity.COMPLEX
        elif word_count > 50 or complex_count > 0:
            return TaskComplexity.MODERATE
        else:
            return TaskComplexity.SIMPLE

    def should_use_simple_execution(self, task_description: str) -> bool:
        """Determine if a task should use simple execution instead of complex planning."""
        complexity = self.analyze_task_complexity(task_description)

        if complexity == TaskComplexity.SIMPLE:
            return True

        single_step_indicators = [
            "explain",
            "describe",
            "what is",
            "how to",
            "define",
            "summarize",
            "tell me about",
            "give me",
            "show me",
            "find",
            "search for",
        ]

        task_lower = task_description.lower()
        if any(indicator in task_lower for indicator in single_step_indicators):
            return True

        return False

    async def simple_task_execution(
        self, task_description: str, config: RunnableConfig = None, **kwargs
    ) -> dict[str, Any]:
        """Execute simple tasks without complex planning."""
        logger.info(f"Executing simple task: {task_description[:100]}...")

        if self.agents:

            selected_agent = self.agents[0]

            try:
                result = await selected_agent.run_async(
                    input_data={"input": task_description},
                    config=config,
                    run_depends=self._run_depends,
                    **kwargs,
                )

                if result.status == RunnableStatus.SUCCESS:
                    return {"content": result.output.get("content")}
                else:
                    return {"content": f"Task failed: {result.error.message}"}

            except Exception as e:
                logger.error(f"Simple task execution failed: {e}")
                return {"content": f"Error executing task: {str(e)}"}
        else:
            return {"content": "No agents available for task execution"}

    def handle_conversational_response(
        self, user_input: str, analysis: ConversationAnalysis, config: RunnableConfig = None, **kwargs
    ) -> dict[str, Any]:
        """
        Handle a conversational response (respond or clarify decisions).

        Args:
            user_input: The user's input
            analysis: The conversation analysis
            config: Configuration for the runnable
            **kwargs: Additional keyword arguments

        Returns:
            Dict containing the conversational response
        """
        if analysis.decision == ConversationDecision.CLARIFY:
            action = "clarify"
        else:
            action = "respond"

        try:

            manager_result = self.manager.run(
                input_data={
                    "action": action,
                    "user_input": user_input,
                    "agents": self.agents_descriptions,
                    "tools": self.tools_descriptions,
                    "chat_history": format_chat_history(self._chat_history),
                    "analysis_reasoning": analysis.reasoning,
                },
                config=config,
                run_depends=self._run_depends,
                **kwargs,
            )
            self._run_depends = [NodeDependency(node=self.manager).to_dict()]

            if manager_result.status == RunnableStatus.SUCCESS:
                response_data = manager_result.output.get("content")
                if isinstance(response_data, dict) and "result" in response_data:
                    response_content = response_data["result"]
                else:
                    response_content = response_data

                self._conversation_context.update(
                    {
                        "last_interaction_type": analysis.decision.value,
                        "last_reasoning": analysis.reasoning,
                        "suggested_next_steps": analysis.suggested_next_steps,
                    }
                )

                return {"content": response_content}
            else:

                if analysis.message:
                    return {"content": analysis.message}
                else:
                    return {"content": "I'm here to help! How can I assist you today?"}

        except Exception as e:
            logger.error(f"Error in conversational response: {e}")

            if analysis.message:
                return {"content": analysis.message}
            else:
                return {"content": "I'd be happy to help! What would you like to work on?"}

    def create_execution_plan(self, input_task: str, config: RunnableConfig = None, **kwargs) -> ParallelExecutionPlan:
        """
        Create an execution plan for parallel task processing.

        Args:
            input_task: The main task to be executed
            config: Configuration for the runnable
            **kwargs: Additional keyword arguments

        Returns:
            ParallelExecutionPlan: The created execution plan
        """
        logger.info(f"Orchestrator {self.name} - {self.id}: Creating execution plan for: {input_task[:100]}...")

        manager_result = self.manager.run(
            input_data={
                "action": "plan",
                "input_task": input_task,
                "agents": self.agents_descriptions,
                "tools": self.tools_descriptions,
                "chat_history": format_chat_history(self._chat_history),
            },
            config=config,
            run_depends=self._run_depends,
            **kwargs,
        )
        self._run_depends = [NodeDependency(node=self.manager).to_dict()]

        if manager_result.status != RunnableStatus.SUCCESS:
            error_message = f"Manager '{self.manager.name}' failed: {manager_result.error.message}"
            raise OrchestratorError(f"Failed to create execution plan: {error_message}")

        planning_data = manager_result.output.get("content")
        if isinstance(planning_data, dict) and "result" in planning_data:
            planning_content = planning_data["result"]
        else:
            planning_content = planning_data
        logger.debug(f"Raw planning content: {planning_content}")

        try:
            plan_data = self._parse_execution_plan(planning_content)
            return ParallelExecutionPlan(**plan_data)
        except Exception as e:
            logger.error(f"Failed to parse execution plan: {e}")
            logger.error(f"Planning content that failed to parse: {planning_content}")

            return self._create_fallback_plan(input_task)

    def _parse_execution_plan(self, planning_content: str) -> dict[str, Any]:
        """Parse the execution plan from manager output."""
        logger.debug(f"Parsing execution plan from: {planning_content[:200]}...")

        json_content = None

        if "<output>" in planning_content and "</output>" in planning_content:
            start = planning_content.find("<output>") + len("<output>")
            end = planning_content.find("</output>")
            json_content = planning_content[start:end].strip()

            json_content = json_content.replace("```json", "").replace("```", "").strip()
        elif "```json" in planning_content:
            start = planning_content.find("```json") + len("```json")
            end = planning_content.find("```", start)
            json_content = planning_content[start:end].strip()
        else:

            start = planning_content.find("{")
            end = planning_content.rfind("}") + 1
            if start != -1 and end > start:
                json_content = planning_content[start:end]
            else:
                logger.error(f"No JSON content found in planning: {planning_content}")
                raise ValueError("No valid output tags found in planning content")

        logger.debug(f"Extracted planning JSON content: {json_content}")

        try:
            return json.loads(json_content)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in planning: {e}")
            logger.error(f"Problematic planning JSON: {json_content}")

            fixed_content = json_content.replace("'", '"').replace("True", "true").replace("False", "false")
            try:
                return json.loads(fixed_content)
            except json.JSONDecodeError as e2:
                logger.error(f"Still failed to parse planning JSON after fixes: {e2}")
                logger.error(f"Final planning JSON content: {fixed_content}")
                raise

    def _create_fallback_plan(self, input_task: str) -> ParallelExecutionPlan:
        """Create a simple fallback execution plan."""
        task = ParallelTask(
            name="Fallback Task",
            description=input_task,
            complexity=TaskComplexity.MODERATE,
            task_type=TaskType.AGENT,
            agent_name=self.agents[0].name if self.agents else None,
        )

        return ParallelExecutionPlan(tasks=[task], execution_groups=[[task.id]], max_concurrency=1)

    async def execute_parallel_plan(
        self, plan: ParallelExecutionPlan, config: RunnableConfig = None, **kwargs
    ) -> dict[str, Any]:
        """
        Execute the parallel execution plan.

        Args:
            plan: The execution plan to run
            config: Configuration for the runnable
            **kwargs: Additional keyword arguments

        Returns:
            Dict containing the execution results
        """
        logger.info(f"Orchestrator {self.name} - {self.id}: Executing parallel plan with {len(plan.tasks)} tasks")

        self._shared_context.update(plan.shared_context)

        for group_idx, task_group in enumerate(plan.execution_groups):
            logger.info(f"Executing group {group_idx + 1}/{len(plan.execution_groups)} with {len(task_group)} tasks")

            group_tasks = [task for task in plan.tasks if task.id in task_group]

            group_results = await self._execute_task_group(group_tasks, config, **kwargs)

            for task_id, result in group_results.items():
                self._execution_results[task_id] = result

                if result.success and result.context_updates:
                    self._shared_context.update(result.context_updates)

                task_obj = next((t for t in group_tasks if t.id == task_id), None)
                if task_obj and result.success:

                    task_result_content = result.result
                    self._update_semantic_context(task_obj, task_result_content)

        return await self._generate_final_result(plan, config, **kwargs)

    async def _execute_task_group(
        self, tasks: list[ParallelTask], config: RunnableConfig = None, **kwargs
    ) -> dict[str, TaskResult]:
        """Execute a group of tasks in parallel."""

        task_coroutines = []
        for task in tasks:
            if task.task_type == TaskType.AGENT:
                coro = self._execute_agent_task(task, config, **kwargs)
            elif task.task_type == TaskType.TOOL:
                coro = self._execute_tool_task(task, config, **kwargs)
            else:
                coro = self._execute_hybrid_task(task, config, **kwargs)

            task_coroutines.append(coro)

        semaphore = asyncio.Semaphore(min(self.max_concurrency, len(tasks)))

        async def execute_with_semaphore(coro):
            async with semaphore:
                return await asyncio.wait_for(coro, timeout=self.task_timeout)

        try:
            results = await asyncio.gather(*[execute_with_semaphore(coro) for coro in task_coroutines])
            return {tasks[i].id: results[i] for i in range(len(tasks))}
        except Exception as e:
            logger.error(f"Error in parallel execution: {e}")

            return {}

    async def _execute_agent_task(self, task: ParallelTask, config: RunnableConfig = None, **kwargs) -> TaskResult:
        """Execute a task using an agent."""
        start_time = asyncio.get_event_loop().time()

        try:

            agent = next((a for a in self.agents if a.name == task.agent_name), None)
            if not agent:
                raise ValueError(f"Agent '{task.agent_name}' not found")

            task_input = self._prepare_task_input(task)

            result = await agent.run_async(
                input_data={"input": task_input},
                config=config,
                run_depends=self._run_depends,
                **kwargs,
            )

            execution_time = asyncio.get_event_loop().time() - start_time

            if result.status == RunnableStatus.SUCCESS:
                return TaskResult(
                    task_id=task.id,
                    success=True,
                    result=result.output.get("content"),
                    execution_time=execution_time,
                    context_updates={f"task_{task.id}_result": result.output.get("content")},
                )
            else:
                return TaskResult(
                    task_id=task.id, success=False, error=result.error.message, execution_time=execution_time
                )

        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            return TaskResult(task_id=task.id, success=False, error=str(e), execution_time=execution_time)

    async def _execute_tool_task(self, task: ParallelTask, config: RunnableConfig = None, **kwargs) -> TaskResult:
        """Execute a task using a tool directly."""
        start_time = asyncio.get_event_loop().time()

        try:

            tool = next((t for t in self.tools if t.name == task.tool_name), None)
            if not tool:
                raise ValueError(f"Tool '{task.tool_name}' not found")

            task_input = self._prepare_task_input(task)

            result = await tool.run_async(
                input_data={"input": task_input},
                config=config,
                run_depends=self._run_depends,
                **kwargs,
            )

            execution_time = asyncio.get_event_loop().time() - start_time

            if result.status == RunnableStatus.SUCCESS:
                return TaskResult(
                    task_id=task.id,
                    success=True,
                    result=result.output.get("content"),
                    execution_time=execution_time,
                    context_updates={f"task_{task.id}_result": result.output.get("content")},
                )
            else:
                return TaskResult(
                    task_id=task.id, success=False, error=result.error.message, execution_time=execution_time
                )

        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            return TaskResult(task_id=task.id, success=False, error=str(e), execution_time=execution_time)

    async def _execute_hybrid_task(self, task: ParallelTask, config: RunnableConfig = None, **kwargs) -> TaskResult:
        """Execute a hybrid task that may use both agents and tools."""

        return await self._execute_agent_task(task, config, **kwargs)

    def _prepare_task_input(self, task: ParallelTask) -> str:
        """Prepare input for a task including relevant context."""
        task_input = task.description

        if self.enable_context_sharing and task.context_requirements:

            context_parts = []
            for req in task.context_requirements:
                if req in self._shared_context:
                    context_parts.append(f"{req}: {self._shared_context[req]}")

            if context_parts:
                task_input += "\n\nRelevant context:\n" + "\n".join(context_parts)

        return task_input

    def _update_semantic_context(self, task: ParallelTask, result_content: Any):
        """Update semantic context keys based on task completion."""

        task_name_lower = task.name.lower()

        context_mappings = {
            "market share": "market_share_data",
            "market data": "market_share_data",
            "financial": "financial_data",
            "competitive": "competitive_data",
            "competition": "competitive_data",
            "sentiment": "sentiment_data",
            "customer": "customer_sentiment",
            "technology": "technology_trends",
            "tech": "technology_trends",
            "predictive": "predictive_analysis",
            "prediction": "predictive_analysis",
            "forecast": "predictive_analysis",
        }

        context_key = None
        for keyword, semantic_key in context_mappings.items():
            if keyword in task_name_lower:
                context_key = semantic_key
                break

        if not context_key:
            context_key = f"task_{task.id}_data"

        self._shared_context[context_key] = result_content
        logger.debug(f"Updated semantic context: {context_key} = {str(result_content)[:100]}...")

    def should_revise_plan(self) -> bool:
        """Determine if the plan should be revised based on current state."""
        if not self.enable_adaptive_planning:
            return False

        if self._plan_revisions >= self.max_plan_revisions:
            return False

        if self._completed_tasks_since_revision >= self.plan_revision_threshold:
            return True

        if len(self._shared_context) > 5:
            return True

        return False

    async def revise_plan(self, original_task: str, config: RunnableConfig = None, **kwargs) -> ParallelExecutionPlan:
        """Revise the execution plan based on current progress and context."""
        logger.info(f"Revising execution plan (revision {self._plan_revisions + 1})")

        completed_tasks = [r for r in self._execution_results.values() if r.success]
        failed_tasks = [r for r in self._execution_results.values() if not r.success]

        progress_summary = f"""
        Original task: {original_task}

        Progress so far:
        - Completed tasks: {len(completed_tasks)}
        - Failed tasks: {len(failed_tasks)}
        - Current context: {list(self._shared_context.keys())}

        Completed task results:
        {[f"Task {r.task_id}: {str(r.result)[:100]}..." for r in completed_tasks[:3]]}

        What remains to be done to complete the original task?
        What new tasks should be added based on the results so far?
        What existing pending tasks should be modified or removed?
        """

        manager_result = await self.manager.run_async(
            input_data={
                "action": "plan",
                "input_task": progress_summary,
                "agents": self.agents_descriptions,
                "tools": self.tools_descriptions,
                "chat_history": format_chat_history(self._chat_history),
                "revision_context": "This is a plan revision - focus on what still needs to be done",
            },
            config=config,
            run_depends=self._run_depends,
            **kwargs,
        )

        if manager_result.status != RunnableStatus.SUCCESS:
            logger.warning(f"Plan revision failed, keeping current plan: {manager_result.error.message}")
            return self._current_plan

        planning_data = manager_result.output.get("content")
        if isinstance(planning_data, dict) and "result" in planning_data:
            planning_content = planning_data["result"]
        else:
            planning_content = planning_data

        try:
            plan_data = self._parse_execution_plan(planning_content)
            revised_plan = ParallelExecutionPlan(**plan_data)

            self._plan_revisions += 1
            self._completed_tasks_since_revision = 0

            logger.info(f"Plan revised successfully - now has {len(revised_plan.tasks)} tasks")
            return revised_plan

        except Exception as e:
            logger.error(f"Failed to parse revised plan: {e}")
            return self._current_plan

    async def adaptive_execute_parallel_plan(
        self, plan: ParallelExecutionPlan, config: RunnableConfig = None, **kwargs
    ) -> dict[str, Any]:
        """Execute the plan with adaptive revision capabilities."""
        logger.info(f"Starting adaptive execution with {len(plan.tasks)} tasks")

        self._current_plan = plan

        self._shared_context.update(plan.shared_context)

        while True:

            next_group = self._find_next_executable_group()
            if not next_group:
                break

            logger.info(f"Executing next group with {len(next_group)} tasks")

            group_tasks = [task for task in self._current_plan.tasks if task.id in next_group]

            group_results = await self._execute_task_group(group_tasks, config, **kwargs)

            for task_id, result in group_results.items():
                self._execution_results[task_id] = result

                if result.success:
                    self._completed_tasks_since_revision += 1

                    if result.context_updates:
                        self._shared_context.update(result.context_updates)

                    task_obj = next((t for t in group_tasks if t.id == task_id), None)
                    if task_obj:
                        self._update_semantic_context(task_obj, result.result)

            if self.should_revise_plan():
                logger.info("Triggering plan revision based on progress")
                revised_plan = await self.revise_plan(self._original_task, config, **kwargs)
                if revised_plan and revised_plan != self._current_plan:
                    self._current_plan = revised_plan
                    logger.info("Plan updated, continuing with revised plan")

        return await self._generate_final_result(self._current_plan, config, **kwargs)

    def _find_next_executable_group(self) -> list[str] | None:
        """Find the next group of tasks that can be executed (dependencies satisfied)."""
        if not self._current_plan:
            return None

        pending_tasks = [task for task in self._current_plan.tasks if task.id not in self._execution_results]

        if not pending_tasks:
            return None

        executable_tasks = []
        for task in pending_tasks:
            dependencies_satisfied = True
            for dep_id in task.dependencies:
                if dep_id not in self._execution_results or not self._execution_results[dep_id].success:
                    dependencies_satisfied = False
                    break

            if dependencies_satisfied:
                executable_tasks.append(task.id)

        if self._current_plan.execution_groups:
            for group in self._current_plan.execution_groups:
                group_executable = [tid for tid in group if tid in executable_tasks]
                if group_executable:
                    return group_executable

        return executable_tasks[: self.max_concurrency] if executable_tasks else None

    async def _generate_final_result(
        self, plan: ParallelExecutionPlan, config: RunnableConfig = None, **kwargs
    ) -> dict[str, Any]:
        """Generate the final result from all task executions."""

        successful_results = {
            task_id: result.result for task_id, result in self._execution_results.items() if result.success
        }

        errors = {task_id: result.error for task_id, result in self._execution_results.items() if not result.success}

        try:
            final_result = await self.manager.run_async(
                input_data={
                    "action": "final",
                    "successful_results": successful_results,
                    "errors": errors,
                    "shared_context": self._shared_context,
                },
                config=config,
                run_depends=self._run_depends,
                **kwargs,
            )

            if final_result.status == RunnableStatus.SUCCESS:
                final_data = final_result.output.get("content")
                if isinstance(final_data, dict) and "result" in final_data:
                    final_content = final_data["result"]
                else:
                    final_content = final_data
                return {"content": final_content}
            else:

                return {"content": self._simple_result_aggregation(successful_results, errors)}

        except Exception as e:
            logger.error(f"Error generating final result: {e}")
            return {"content": self._simple_result_aggregation(successful_results, errors)}

    def _simple_result_aggregation(self, results: dict[str, Any], errors: dict[str, str]) -> str:
        """Simple fallback for result aggregation."""
        if not results and not errors:
            return "No results generated."

        output_parts = []

        if results:
            output_parts.append("Successfully completed tasks:")
            for task_id, result in results.items():
                output_parts.append(f"- Task {task_id}: {str(result)[:200]}...")

        if errors:
            output_parts.append("\nErrors encountered:")
            for task_id, error in errors.items():
                output_parts.append(f"- Task {task_id}: {error}")

        return "\n".join(output_parts)

    def run_flow(self, input_task: str, config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        """
        Process the given task using conversational intelligence and parallel execution.

        Args:
            input_task: The task to be processed
            config: Configuration for the runnable
            **kwargs: Additional keyword arguments

        Returns:
            Dict containing the final output
        """
        logger.info(
            f"Orchestrator {self.name} - {self.id}: Processing input with conversation mode: {self.conversation_mode}"
        )

        self._chat_history.append({"role": "user", "content": input_task})

        if self.conversation_mode:

            analysis = self.analyze_conversation(input_task, config, **kwargs)

            logger.info(f"Conversation analysis: {analysis.decision.value} (confidence: {analysis.confidence})")

            if analysis.decision in [ConversationDecision.RESPOND, ConversationDecision.CLARIFY]:

                result = self.handle_conversational_response(input_task, analysis, config, **kwargs)
            else:

                result = self._handle_orchestrated_task(input_task, config, **kwargs)
        else:

            user_analysis = self._analyze_user_input(
                input_task,
                f"Agents: {self.agents_descriptions}\nTools: {self.tools_descriptions}",
                config=config,
                **kwargs,
            )

            if user_analysis.decision == Decision.RESPOND:
                result = {"content": user_analysis.message}
            else:
                result = self._handle_orchestrated_task(input_task, config, **kwargs)

        if result and "content" in result:
            self._chat_history.append({"role": "assistant", "content": result["content"]})

        return result

    def _handle_orchestrated_task(self, input_task: str, config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        """Handle a task that requires orchestration."""

        self._original_task = input_task

        if self.should_use_simple_execution(input_task):
            logger.info("Using simple execution for this task")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self.simple_task_execution(input_task, config, **kwargs))
                return result
            finally:
                loop.close()

        plan = self.create_execution_plan(input_task, config, **kwargs)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            if self.enable_adaptive_planning:
                result = loop.run_until_complete(self.adaptive_execute_parallel_plan(plan, config, **kwargs))
            else:
                result = loop.run_until_complete(self.execute_parallel_plan(plan, config, **kwargs))
            return result
        finally:
            loop.close()

    def setup_streaming(self) -> None:
        """Set up streaming for orchestrator."""
        self.manager.streaming = self.streaming
        for agent in self.agents:
            agent.streaming = self.streaming
        for tool in self.tools:
            if hasattr(tool, "streaming"):
                tool.streaming = self.streaming
