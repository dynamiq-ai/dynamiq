import json
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import cached_property
from importlib.metadata import distributions
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.callbacks import BaseCallbackHandler
from dynamiq.callbacks.base import get_execution_run_id, get_parent_run_id, get_run_id
from dynamiq.clients import BaseTracingClient
from dynamiq.utils import JsonWorkflowEncoder, format_value, generate_uuid

UTC = timezone.utc


class RunStatus(str, Enum):
    """Enumeration for run statuses."""
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class RunType(str, Enum):
    """Enumeration for run types."""
    WORKFLOW = "workflow"
    FLOW = "flow"
    NODE = "node"


@dataclass
class ExecutionRun:
    """Data class for execution run details.

    Attributes:
        id (UUID): Execution run ID.
        start_time (datetime): Start time of the execution.
        end_time (datetime | None): End time of the execution.
        status (RunStatus | None): Status of the execution.
        input (Any | None): Input data for the execution.
        output (Any | None): Output data from the execution.
        error (Any | None): Error details if any.
        metadata (dict): Additional metadata.
    """
    id: UUID
    start_time: datetime
    end_time: datetime | None = None
    status: RunStatus | None = None
    input: Any | None = None
    output: Any | None = None
    error: Any | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert ExecutionRun to dictionary.

        Returns:
            dict: Dictionary representation of ExecutionRun.
        """
        return asdict(self)


@dataclass
class Run:
    """Data class for run details.

    Attributes:
        id (UUID): Run ID.
        name (str): Name of the run.
        type (RunType): Type of the run.
        trace_id (UUID | str): Trace ID.
        source_id (UUID | str): Source ID.
        session_id (UUID | str): Session ID.
        start_time (datetime): Start time of the run.
        end_time (datetime): End time of the run.
        parent_run_id (UUID): Parent run ID.
        status (RunStatus): Status of the run.
        input (Any): Input data for the run.
        output (Any): Output data from the run.
        metadata (Any): Additional metadata.
        error (Any): Error details if any.
        executions (list[ExecutionRun]): List of execution runs.
        tags (list[str]): List of tags.
    """
    id: UUID
    name: str
    type: RunType
    trace_id: UUID | str
    source_id: UUID | str
    session_id: UUID | str
    start_time: datetime
    end_time: datetime = None
    parent_run_id: UUID = None
    status: RunStatus = None
    input: Any = None
    output: Any = None
    metadata: Any = None
    error: Any = None
    executions: list[ExecutionRun] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert Run to dictionary.

        Returns:
            dict: Dictionary representation of Run.
        """
        return asdict(self)

    def to_json(self) -> str:
        """Convert Run to JSON string.

        Returns:
            str: JSON string representation of Run.
        """
        return json.dumps(self.to_dict(), cls=JsonWorkflowEncoder)


class TracingCallbackHandler(BaseModel, BaseCallbackHandler):
    """Callback handler for tracing workflow events.

    Attributes:
        source_id (str | None): Source ID.
        trace_id (str | None): Trace ID.
        session_id (str | None): Session ID.
        client (BaseTracingClient | None): Tracing client.
        runs (dict[UUID, Run]): Dictionary of runs.
        tags (list[str]): List of tags.
        installed_pkgs (list[str]): List of installed packages.
    """
    source_id: str | None = Field(default_factory=generate_uuid)
    trace_id: str | None = Field(default_factory=generate_uuid)
    session_id: str | None = Field(default_factory=generate_uuid)
    client: BaseTracingClient | None = None
    runs: dict[UUID, Run] = {}
    tags: list[str] = []
    metadata: dict = {}

    installed_pkgs: list[str] = Field(
        ["dynamiq"],
        description="List of installed packages to include in the host information.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @cached_property
    def host(self) -> dict:
        """Get host information.

        Returns:
            dict: Host information including installed packages.
        """
        return {
            "installed_pkgs": [
                {"name": dist.metadata["Name"], "version": dist.version}
                for dist in distributions()
                if dist.metadata.get("Name") in self.installed_pkgs
            ],
        }

    def _get_node_base_run(self, serialized: dict[str, Any], **kwargs: Any) -> Run:
        """Get base run details for a node.

        Args:
            serialized (dict[str, Any]): Serialized node data.
            **kwargs (Any): Additional arguments.

        Returns:
            Run: Base run details for the node.
        """
        run_id = get_run_id(kwargs)
        parent_run_id = get_parent_run_id(kwargs)

        from dynamiq.nodes import NodeGroup

        # Handle runtime LLM prompt override
        if serialized.get("group") == NodeGroup.LLMS:
            prompt = kwargs.get("prompt") or serialized.get("prompt")
            if isinstance(prompt, BaseModel):
                prompt = prompt.model_dump()
            serialized["prompt"] = prompt

        truncate_metadata = {}
        formatted_input, truncate_metadata.setdefault("truncated", {})["input"] = format_value(
            kwargs.get("input_data"), truncate_enabled=True
        )
        run = Run(
            id=run_id,
            name=serialized.get("name"),
            type=RunType.NODE,
            trace_id=self.trace_id,
            source_id=self.source_id,
            session_id=self.session_id,
            start_time=datetime.now(UTC),
            parent_run_id=parent_run_id,
            metadata={
                "node": serialized,
                "run_depends": kwargs.get("run_depends", []),
                "host": self.host,
                **self.metadata,
                **truncate_metadata,
            },
            tags=self.tags,
            input=formatted_input,
        )
        return run

    def on_workflow_start(
        self, serialized: dict[str, Any], input_data: dict[str, Any], **kwargs: Any
    ):
        """Called when the workflow starts.

        Args:
            serialized (dict[str, Any]): Serialized workflow data.
            input_data (dict[str, Any]): Input data for the workflow.
            **kwargs (Any): Additional arguments.
        """
        run_id = get_run_id(kwargs)

        truncate_metadata = {}
        formatted_input, truncate_metadata.setdefault("truncated", {})["input"] = format_value(
            input_data, truncate_enabled=True
        )
        self.runs[run_id] = Run(
            id=run_id,
            name="Workflow",
            type=RunType.WORKFLOW,
            trace_id=self.trace_id,
            source_id=self.source_id,
            session_id=self.session_id,
            start_time=datetime.now(UTC),
            input=formatted_input,
            metadata={
                "workflow": {"id": serialized.get("id"), "version": serialized.get("version")},
                "host": self.host,
                **self.metadata,
                **truncate_metadata,
            },
            tags=self.tags,
        )

    def on_workflow_end(
        self, serialized: dict[str, Any], output_data: dict[str, Any], **kwargs: Any
    ):
        """Called when the workflow ends.

        Args:
            serialized (dict[str, Any]): Serialized workflow data.
            output_data (dict[str, Any]): Output data from the workflow.
            **kwargs (Any): Additional arguments.
        """
        run = ensure_run(get_run_id(kwargs), self.runs)
        run.end_time = datetime.now(UTC)
        run.output, run.metadata.setdefault("truncated", {})["output"] = format_value(
            output_data, truncate_enabled=True
        )
        run.status = RunStatus.SUCCEEDED

        self.flush()

    def on_workflow_error(
        self, serialized: dict[str, Any], error: BaseException, **kwargs: Any
    ):
        """Called when the workflow errors.

        Args:
            serialized (dict[str, Any]): Serialized workflow data.
            error (BaseException): Error encountered.
            **kwargs (Any): Additional arguments.
        """
        run = ensure_run(get_run_id(kwargs), self.runs)
        run.end_time = datetime.now(UTC)
        run.status = RunStatus.FAILED
        run.error = {
            "message": str(error),
            "traceback": traceback.format_exc(),
        }

    def on_flow_start(
        self, serialized: dict[str, Any], input_data: dict[str, Any], **kwargs: Any
    ):
        """Called when the flow starts.

        Args:
            serialized (dict[str, Any]): Serialized flow data.
            input_data (dict[str, Any]): Input data for the flow.
            **kwargs (Any): Additional arguments.
        """
        run_id = get_run_id(kwargs)
        parent_run_id = get_parent_run_id(kwargs)
        truncate_metadata = {}
        formatted_input, truncate_metadata.setdefault("truncated", {})["input"] = format_value(
            input_data, truncate_enabled=True
        )

        self.runs[run_id] = Run(
            id=run_id,
            name="Flow",
            type=RunType.FLOW,
            trace_id=self.trace_id,
            source_id=self.source_id,
            session_id=self.session_id,
            start_time=datetime.now(UTC),
            parent_run_id=parent_run_id,
            input=formatted_input,
            metadata={"flow": {"id": serialized.get("id")}, "host": self.host, **self.metadata, **truncate_metadata},
            tags=self.tags,
        )

    def on_flow_end(
        self, serialized: dict[str, Any], output_data: dict[str, Any], **kwargs: Any
    ):
        """Called when the flow ends.

        Args:
            serialized (dict[str, Any]): Serialized flow data.
            output_data (dict[str, Any]): Output data from the flow.
            **kwargs (Any): Additional arguments.
        """
        run = ensure_run(get_run_id(kwargs), self.runs)
        run.end_time = datetime.now(UTC)
        run.output, run.metadata.setdefault("truncated", {})["output"] = format_value(
            output_data, truncate_enabled=True
        )
        run.status = RunStatus.SUCCEEDED

    def on_flow_error(
        self, serialized: dict[str, Any], error: BaseException, **kwargs: Any
    ):
        """Called when the flow errors.

        Args:
            serialized (dict[str, Any]): Serialized flow data.
            error (BaseException): Error encountered.
            **kwargs (Any): Additional arguments.
        """
        run = ensure_run(get_run_id(kwargs), self.runs)
        run.end_time = datetime.now(UTC)
        run.status = RunStatus.FAILED
        run.error = {
            "message": str(error),
            "traceback": traceback.format_exc(),
        }

    def on_node_start(
        self, serialized: dict[str, Any], input_data: dict[str, Any], **kwargs: Any
    ):
        """Called when the node starts.

        Args:
            serialized (dict[str, Any]): Serialized node data.
            input_data (dict[str, Any]): Input data for the node.
            **kwargs (Any): Additional arguments.
        """
        run_id = get_run_id(kwargs)
        run = self._get_node_base_run(serialized, **kwargs)
        run.input, run.metadata.setdefault("truncated", {})["input"] = format_value(input_data, truncate_enabled=True)
        self.runs[run_id] = run

    def on_node_end(
        self, serialized: dict[str, Any], output_data: dict[str, Any], **kwargs: Any
    ):
        """Called when the node ends.

        Args:
            serialized (dict[str, Any]): Serialized node data.
            output_data (dict[str, Any]): Output data from the node.
            **kwargs (Any): Additional arguments.
        """
        run = ensure_run(get_run_id(kwargs), self.runs)
        run.end_time = datetime.now(UTC)
        run.output, run.metadata.setdefault("truncated", {})["output"] = format_value(
            output_data, truncate_enabled=True
        )
        run.status = RunStatus.SUCCEEDED
        run.metadata["is_output_from_cache"] = kwargs.get("is_output_from_cache", False)

    def on_node_error(
        self, serialized: dict[str, Any], error: BaseException, **kwargs: Any
    ):
        """Called when the node errors.

        Args:
            serialized (dict[str, Any]): Serialized node data.
            error (BaseException): Error encountered.
            **kwargs (Any): Additional arguments.
        """
        run_id = get_run_id(kwargs)
        if (run := self.runs.get(run_id)) is None:
            run = self._get_node_base_run(serialized, **kwargs)
            self.runs[run_id] = run

        run.end_time = datetime.now(UTC)
        run.status = RunStatus.FAILED
        run.error = {
            "message": str(error),
            "traceback": traceback.format_exc(),
        }

    def on_node_skip(
        self,
        serialized: dict[str, Any],
        skip_data: dict[str, Any],
        input_data: dict[str, Any],
        **kwargs: Any,
    ):
        """Called when the node skips.

        Args:
            serialized (dict[str, Any]): Serialized node data.
            skip_data (dict[str, Any]): Data related to the skip.
            input_data (dict[str, Any]): Input data for the node.
            **kwargs (Any): Additional arguments.
        """
        run_id = get_run_id(kwargs)
        if (run := self.runs.get(run_id)) is None:
            run = self._get_node_base_run(serialized, **kwargs)
            self.runs[run_id] = run

        run.input, run.metadata.setdefault("truncated", {})["input"] = format_value(input_data, truncate_enabled=True)
        run.end_time = run.start_time
        run.status = RunStatus.SKIPPED
        run.metadata["skip"] = format_value(skip_data)[0]

    def on_node_execute_start(
        self, serialized: dict[str, Any], input_data: dict[str, Any], **kwargs: Any
    ):
        """Called when the node execute starts.

        Args:
            serialized (dict[str, Any]): Serialized node data.
            input_data (dict[str, Any]): Input data for the node.
            **kwargs (Any): Additional arguments.
        """
        run = ensure_run(get_run_id(kwargs), self.runs)
        execution_run_id = get_execution_run_id(kwargs)
        truncate_metadata = {}
        formatted_input, truncate_metadata.setdefault("truncated", {})["input"] = format_value(
            input_data, truncate_enabled=True
        )
        execution = ExecutionRun(
            id=execution_run_id,
            start_time=datetime.now(UTC),
            input=formatted_input,
            metadata=truncate_metadata,
        )
        run.executions.append(execution)

    def on_node_execute_end(
        self, serialized: dict[str, Any], output_data: dict[str, Any], **kwargs: Any
    ):
        """Called when the node execute ends.

        Args:
            serialized (dict[str, Any]): Serialized node data.
            output_data (dict[str, Any]): Output data from the node.
            **kwargs (Any): Additional arguments.
        """
        run = ensure_run(get_run_id(kwargs), self.runs)
        execution = ensure_execution_run(get_execution_run_id(kwargs), run.executions)
        execution.end_time = datetime.now(UTC)
        execution.output, execution.metadata.setdefault("truncated", {})["output"] = format_value(
            output_data, truncate_enabled=True
        )
        execution.status = RunStatus.SUCCEEDED

    def on_node_execute_error(
        self, serialized: dict[str, Any], error: BaseException, **kwargs: Any
    ):
        """Called when the node execute errors.

        Args:
            serialized (dict[str, Any]): Serialized node data.
            error (BaseException): Error encountered.
            **kwargs (Any): Additional arguments.
        """
        run = ensure_run(get_run_id(kwargs), self.runs)
        execution = ensure_execution_run(get_execution_run_id(kwargs), run.executions)
        execution.end_time = datetime.now(UTC)
        execution.status = RunStatus.FAILED
        execution.error = {
            "message": str(error),
            "traceback": traceback.format_exc(),
        }

    def on_node_execute_run(self, serialized: dict[str, Any], **kwargs: Any):
        """Called when the node execute runs.

        Args:
            serialized (dict[str, Any]): Serialized node data.
            **kwargs (Any): Additional arguments.
        """
        run = ensure_run(get_run_id(kwargs), self.runs)
        if usage := kwargs.get("usage_data"):
            run.metadata["usage"] = usage

        if prompt_messages := kwargs.get("prompt_messages"):
            run.metadata["node"]["prompt"]["messages"] = prompt_messages

    def flush(self):
        """Flush the runs to the tracing client."""
        if self.client:
            self.client.trace([run for run in self.runs.values()])


def ensure_run(run_id: UUID, runs: dict[UUID, Run]) -> Run:
    """Ensure the run exists in the runs dictionary.

    Args:
        run_id (UUID): Run ID.
        runs (dict[UUID, Run]): Dictionary of runs.

    Returns:
        Run: The run corresponding to the run ID.

    Raises:
        ValueError: If the run is not found.
    """
    run = runs.get(run_id)
    if not run:
        raise ValueError(f"run {run_id} not found")

    return runs[run_id]


def ensure_execution_run(execution_run_id: UUID, executions: list[ExecutionRun]) -> ExecutionRun:
    """Ensure the execution run exists in the executions list.

    Args:
        execution_run_id (UUID): Execution run ID.
        executions (list[ExecutionRun]): List of execution runs.

    Returns:
        ExecutionRun: The execution run corresponding to the execution run ID.

    Raises:
        ValueError: If the execution run is not found.
    """
    for execution in executions:
        if execution.id == execution_run_id:
            return execution

    raise ValueError(f"execution run {execution_run_id} not found")
