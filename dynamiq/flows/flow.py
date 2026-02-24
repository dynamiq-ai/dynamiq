import asyncio
import builtins
import time
from datetime import datetime
from functools import cached_property
from graphlib import CycleError, TopologicalSorter
from io import BytesIO
from typing import Any
from uuid import uuid4

from pydantic import Field, PrivateAttr, computed_field, field_validator

from dynamiq.checkpoints.checkpoint import (
    CheckpointConfig,
    CheckpointContext,
    CheckpointMixin,
    CheckpointStatus,
    FlowCheckpoint,
    NodeCheckpointState,
    PendingInputContext,
)
from dynamiq.connections.managers import ConnectionManager
from dynamiq.executors.base import BaseExecutor
from dynamiq.executors.pool import ThreadExecutor
from dynamiq.flows.base import BaseFlow
from dynamiq.nodes.node import Node, NodeReadyToRun
from dynamiq.nodes.types import Behavior
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus
from dynamiq.runnables.base import RunnableFailedNodeInfo, RunnableResultError
from dynamiq.utils.duration import format_duration
from dynamiq.utils.logger import logger


class FlowNodeFailureException(Exception):
    """Exception raised when one or more nodes with RAISE behavior failed during flow execution."""

    def __init__(self, message: str, failed_nodes: list[RunnableFailedNodeInfo] | None = None):
        super().__init__(message)
        self.failed_nodes = failed_nodes or []


class Flow(BaseFlow):
    """
    A class for managing and executing a graph-like structure of nodes.

    Attributes:
        nodes (list[Node]): List of nodes in the flow.
        executor (type[BaseExecutor]): Executor class for running nodes. Defaults to ThreadExecutor.
        max_node_workers (int | None): Maximum number of concurrent node workers. Defaults to None.
        connection_manager (ConnectionManager): Manager for handling connections. Defaults to ConnectionManager().
        checkpoint (CheckpointConfig): Flow-level checkpoint defaults (backend, retention, behavior).

    Checkpointing uses a two-layer config:
    - Flow-level (CheckpointConfig): structural defaults - backend, retention, default behavior.
    - Run-level (CheckpointRunConfig via RunnableConfig): per-run overrides - enabled, resume_from, exclude_node_ids.

    Example:
        >>> from dynamiq.checkpoints.backends.filesystem import FileSystem
        >>>
        >>> # Define flow with backend defaults
        >>> flow = Flow(
        ...     nodes=[agent1, agent2],
        ...     checkpoint=CheckpointConfig(enabled=True, backend=FileSystem(base_path=".checkpoints")),
        ... )
        >>>
        >>> # Run with checkpointing (uses flow defaults)
        >>> result = flow.run_sync(input_data={"query": "..."})
        >>>
        >>> # Resume via RunnableConfig (preferred)
        >>> config = RunnableConfig(checkpoint=CheckpointConfig(resume_from=checkpoint_id))
        >>> result = flow.run_sync(input_data=None, config=config)
        >>>
        >>> # Resume via kwarg (backward compatible)
        >>> result = flow.run_sync(input_data=None, resume_from=checkpoint_id)
    """

    name: str = "Flow"
    nodes: list[Node] = []
    executor: type[BaseExecutor] = ThreadExecutor
    max_node_workers: int | None = None
    connection_manager: ConnectionManager = Field(default_factory=ConnectionManager)

    checkpoint: CheckpointConfig = Field(
        default_factory=CheckpointConfig, description="Configuration for checkpoint/resume functionality"
    )

    # Private attributes for checkpoint state
    _checkpoint: FlowCheckpoint | None = PrivateAttr(default=None)
    _effective_checkpoint_config: CheckpointConfig | None = PrivateAttr(default=None)
    _original_input: Any = PrivateAttr(default=None)

    def __init__(self, **kwargs):
        """
        Initializes the Flow instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self._node_by_id = {node.id: node for node in self.nodes}
        self._ts = None

        self._init_components()
        self.reset_run_state()

    @computed_field
    @cached_property
    def type(self) -> str:
        return f"{self.__module__.rsplit('.', 1)[0]}.{self.__class__.__name__}"

    @property
    def to_dict_exclude_params(self):
        return {"nodes": True, "connection_manager": True, "checkpoint": True}

    def to_dict(self, include_secure_params: bool = True, for_tracing=False, **kwargs) -> dict:
        """Converts the instance to a dictionary.

        Returns:
            dict: A dictionary representation of the instance.
        """
        data = super().to_dict(include_secure_params=include_secure_params, **kwargs)
        data["nodes"] = [
            node.to_dict(include_secure_params=include_secure_params, for_tracing=for_tracing, **kwargs)
            for node in self.nodes
        ]
        return data

    @field_validator("nodes")
    @classmethod
    def validate_nodes(cls, nodes: list[Node]) -> list[Node]:
        """
        Validates the list of nodes in the flow.

        Args:
            nodes (list[Node]): List of nodes to validate.

        Returns:
            list[Node]: Validated list of nodes.

        Raises:
            ValueError: If there are duplicate node IDs or invalid dependencies.
        """
        nodes_ids_unique = set()
        nodes_deps_ids_unique = set()
        for node in nodes:
            if node.id in nodes_ids_unique:
                raise ValueError(
                    f"Flow has nodes with duplicated ids: '{node.id}'. Node ids must be unique."
                )

            nodes_ids_unique.add(node.id)
            node_deps_ids = [dep.node.id for dep in node.depends]
            if len(set(node_deps_ids)) != len(node_deps_ids):
                raise ValueError(
                    f"Flow node '{node.id}' has duplicated dependency ids. Node dependencies ids must be unique."
                )

            nodes_deps_ids_unique.update(node_deps_ids)

        if not nodes_deps_ids_unique.issubset(nodes_ids_unique):
            raise ValueError(
                "Flow nodes have dependencies that are not present in the flow."
            )

        return nodes

    def _init_components(self):
        """Initializes components for nodes with postponed initialization."""
        for node in self.nodes:
            if node.is_postponed_component_init:
                node.init_components(self.connection_manager)

    def _get_nodes_ready_to_run(self, input_data: Any) -> list[NodeReadyToRun]:
        """
        Gets the list of nodes that are ready to run.

        Args:
            input_data (Any): Input data for the nodes.

        Returns:
            list[NodeReadyToRun]: List of nodes ready to run.
        """
        ready_ts_nodes = self._ts.get_ready()
        ready_nodes = []

        completed_result = {
            node_id: result for node_id, result in self._results.items() if result.status != RunnableStatus.UNDEFINED
        }

        for node_id in ready_ts_nodes:
            node = self._node_by_id[node_id]
            is_ready = True
            for dep in node.depends:
                if dep.node.id not in completed_result:
                    is_ready = False
                    break

            ready_node = NodeReadyToRun(
                node=node,
                is_ready=is_ready,
                input_data=input_data,
                depends_result=completed_result,
            )
            ready_nodes.append(ready_node)

        return ready_nodes

    def _get_output(self) -> dict[str, dict]:
        """
        Gets the output of the flow.

        Returns:
            dict[str, dict]: Output of the flow.
        """
        return {
            node_id: result.to_dict(skip_format_types={BytesIO, bytes}) for node_id, result in self._results.items()
        }

    def _get_failed_nodes_with_raise_behavior(self) -> list[RunnableFailedNodeInfo]:
        """
        Gets the list of nodes that failed with RAISE error behavior.

        Returns:
            list[FailedNodeInfo]: List of failed node information.
        """
        failed_nodes: list[RunnableFailedNodeInfo] = []
        for node_id, result in self._results.items():
            node = self._node_by_id.get(node_id)
            if node and result.status == RunnableStatus.FAILURE and node.error_handling.behavior == Behavior.RAISE:
                error_message = result.error.message if result.error else None
                failed_nodes.append(RunnableFailedNodeInfo(id=node_id, name=node.name, error_message=error_message))
        return failed_nodes

    @staticmethod
    def init_node_topological_sorter(nodes: list[Node]):
        """
        Initializes a topological sorter for the given nodes.

        Args:
            nodes (list[Node]): List of nodes to sort.

        Returns:
            TopologicalSorter: Initialized topological sorter.

        Raises:
            CycleError: If a cycle is detected in node dependencies.
        """
        topological_sorter = TopologicalSorter()
        for node in nodes:
            topological_sorter.add(node.id, *[d.node.id for d in node.depends])

        try:
            topological_sorter.prepare()
        except CycleError as e:
            logger.error(f"Node dependencies cycle detected. Error: {e}")
            raise

        return topological_sorter

    def reset_run_state(self):
        """Resets the run state of the flow."""
        self._results = {
            node.id: RunnableResult(status=RunnableStatus.UNDEFINED)
            for node in self.nodes
        }
        self._ts = self.init_node_topological_sorter(nodes=self.nodes)

    def _cleanup_dry_run(self, config: RunnableConfig = None):
        """
        Clean up resources created during dry run.

        Args:
            config (RunnableConfig, optional): Configuration for the run.
        """
        if not config or not getattr(config.dry_run, "enabled", False):
            return

        logger.debug("Starting dry run cleanup...")

        # Filter nodes that have dry_run_cleanup implemented
        nodes_with_cleanup = [
            node
            for node in self.nodes
            if hasattr(node, "dry_run_cleanup")
            and getattr(node, "dry_run_cleanup").__qualname__ != "Node.dry_run_cleanup"
        ]
        logger.debug(f"Nodes with cleanup: {[node.name for node in nodes_with_cleanup]}")

        for node in nodes_with_cleanup:
            try:
                node.dry_run_cleanup(config.dry_run)
            except Exception as e:
                logger.error(f"Failed to clean up dry run resources for node {node.id}: {str(e)}")

    async def _cleanup_dry_run_async(self, config: RunnableConfig = None):
        """Async variant of dry-run cleanup. Runs synchronous cleanup functions in a thread."""
        if not config or not getattr(config.dry_run, "enabled", False):
            return

        logger.debug("Starting async dry run cleanup...")

        # Filter nodes that have dry_run_cleanup implemented
        nodes_with_cleanup = [
            node
            for node in self.nodes
            if hasattr(node, "dry_run_cleanup")
            and getattr(node, "dry_run_cleanup").__qualname__ != "Node.dry_run_cleanup"
        ]
        logger.debug(f"Nodes with cleanup: {[node.name for node in nodes_with_cleanup]}")

        tasks = [asyncio.to_thread(getattr(node, "dry_run_cleanup"), config.dry_run) for node in nodes_with_cleanup]

        if not tasks:
            return

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for node, res in zip(nodes_with_cleanup, results):
            if isinstance(res, Exception):
                logger.error(f"Failed to clean up dry run resources for node {node.id}: {res}")

    def run_sync(
        self,
        input_data: Any,
        config: RunnableConfig = None,
        *,
        resume_from: str | FlowCheckpoint | None = None,
        **kwargs,
    ) -> RunnableResult:
        """
        Run the flow synchronously with the given input data and configuration.

        Args:
            input_data (Any): Input data for the flow. If resuming, this can be None
                              to use the checkpoint's original_input.
            config (RunnableConfig, optional): Configuration for the run. Defaults to None.
            resume_from: Checkpoint ID or FlowCheckpoint to resume from (backward compat).
                         Prefer using config.checkpoint.resume_from instead.
            **kwargs: Additional keyword arguments.

        Returns:
            RunnableResult: Result of the flow execution.

        Raises:
            ValueError: If resume_from is provided but checkpoint not found
        """
        self._effective_checkpoint_config = self._get_effective_checkpoint_config(config)
        effective_resume = resume_from or self._effective_checkpoint_config.resume_from

        if effective_resume:
            self._checkpoint = self._load_checkpoint(effective_resume)
            if not self._checkpoint:
                raise ValueError(f"Checkpoint not found: {effective_resume}")

            self._restore_from_checkpoint(self._checkpoint)

            if input_data is None:
                input_data = self._checkpoint.original_input

            kwargs["resumed_from_checkpoint"] = self._checkpoint.id
            kwargs["original_run_id"] = self._checkpoint.run_id

            logger.info(
                f"Flow {self.id}: resuming from checkpoint {self._checkpoint.id}, "
                f"skipping {len(self._checkpoint.completed_node_ids)} completed nodes"
            )
        else:
            self.reset_run_state()
            self._checkpoint = None

        self._original_input = input_data

        run_id = uuid4()
        wf_run_id = str(config.run_id) if config and config.run_id else str(run_id)
        merged_kwargs = kwargs | {
            "run_id": run_id,
            "parent_run_id": kwargs.get("parent_run_id", None),
        }

        if self._should_checkpoint():
            if self._checkpoint:
                self._checkpoint.run_id = str(run_id)
                self._checkpoint.wf_run_id = wf_run_id
                self._checkpoint.status = CheckpointStatus.ACTIVE
                self._save_checkpoint()
            else:
                self._checkpoint = FlowCheckpoint(
                    flow_id=self.id,
                    run_id=str(run_id),
                    wf_run_id=wf_run_id,
                    original_input=input_data,
                    original_config=config.model_dump() if config else None,
                )
                self._save_checkpoint()

        config = self._setup_checkpoint_context(config)

        logger.info(f"Flow {self.id}: execution started.")
        self.run_on_flow_start(input_data, config, **merged_kwargs)
        time_start = datetime.now()

        try:
            if self.nodes:
                max_workers = (
                    config.max_node_workers if config and config.max_node_workers is not None else self.max_node_workers
                )
                run_executor = self.executor(max_workers=max_workers)

                while self._ts.is_active():
                    ready_nodes = self._get_nodes_ready_to_run(input_data=input_data)

                    if self._checkpoint:
                        already_completed = [
                            n.node.id for n in ready_nodes if n.node.id in self._checkpoint.completed_node_ids
                        ]
                        if already_completed:
                            self._ts.done(*already_completed)
                        ready_nodes = [n for n in ready_nodes if n.node.id not in self._checkpoint.completed_node_ids]

                    if not ready_nodes:
                        time.sleep(0.003)
                        continue

                    results = run_executor.execute(
                        ready_nodes=ready_nodes,
                        config=config,
                        **(merged_kwargs | {"parent_run_id": run_id}),
                    )
                    self._results.update(results)
                    self._ts.done(*results.keys())

                    if self._should_checkpoint():
                        self._update_checkpoint(results, CheckpointStatus.ACTIVE)

                    time.sleep(0.003)

                run_executor.shutdown()

            output = self._get_output()
            failed_nodes = self._get_failed_nodes_with_raise_behavior()

            if failed_nodes:
                if self._checkpoint and self._should_checkpoint_on_failure():
                    self._update_checkpoint({}, CheckpointStatus.FAILED)
                    logger.info(f"Flow {self.id}: checkpoint saved on failure, checkpoint_id={self._checkpoint.id}")

                failed_names = [node.name or node.id for node in failed_nodes]
                error_msg = f"Flow execution failed due to node failures: {', '.join(failed_names)}"
                error = FlowNodeFailureException(error_msg, failed_nodes)
                self.run_on_flow_error(error, config, failed_nodes=failed_nodes, **merged_kwargs)
                logger.error(f"Flow {self.id}: execution failed in {format_duration(time_start, datetime.now())}.")
                return RunnableResult(
                    status=RunnableStatus.FAILURE,
                    input=input_data,
                    output=output,
                    error=RunnableResultError.from_exception(error, failed_nodes=failed_nodes),
                )

            if self._should_checkpoint():
                self._update_checkpoint({}, CheckpointStatus.COMPLETED)
                self._cleanup_old_checkpoints()

            self.run_on_flow_end(output, config, **merged_kwargs)
            logger.info(f"Flow {self.id}: execution succeeded in {format_duration(time_start, datetime.now())}.")
            return RunnableResult(status=RunnableStatus.SUCCESS, input=input_data, output=output)
        except Exception as e:
            if self._checkpoint and self._should_checkpoint_on_failure():
                self._update_checkpoint({}, CheckpointStatus.FAILED)
                logger.info(f"Flow {self.id}: checkpoint saved on failure, checkpoint_id={self._checkpoint.id}")

            failed_nodes = self._get_failed_nodes_with_raise_behavior()
            self.run_on_flow_error(e, config, failed_nodes=failed_nodes, **merged_kwargs)
            logger.error(f"Flow {self.id}: execution failed in {format_duration(time_start, datetime.now())}.")
            return RunnableResult(
                status=RunnableStatus.FAILURE,
                input=input_data,
                error=RunnableResultError.from_exception(e, failed_nodes=failed_nodes),
            )
        finally:
            self._cleanup_dry_run(config)

    async def run_async(
        self,
        input_data: Any,
        config: RunnableConfig = None,
        *,
        resume_from: str | FlowCheckpoint | None = None,
        **kwargs,
    ) -> RunnableResult:
        """
        Run the flow asynchronously with the given input data and configuration.

        Args:
            input_data (Any): Input data for the flow. If resuming, this can be None
                              to use the checkpoint's original_input.
            config (RunnableConfig, optional): Configuration for the run. Defaults to None.
            resume_from: Checkpoint ID or FlowCheckpoint to resume from (backward compat).
                         Prefer using config.checkpoint.resume_from instead.
            **kwargs: Additional keyword arguments.

        Returns:
            RunnableResult: Result of the flow execution.
        """
        self._effective_checkpoint_config = self._get_effective_checkpoint_config(config)
        effective_resume = resume_from or self._effective_checkpoint_config.resume_from

        if effective_resume:
            self._checkpoint = self._load_checkpoint(effective_resume)
            if not self._checkpoint:
                raise ValueError(f"Checkpoint not found: {effective_resume}")

            self._restore_from_checkpoint(self._checkpoint)

            if input_data is None:
                input_data = self._checkpoint.original_input

            kwargs["resumed_from_checkpoint"] = self._checkpoint.id
            kwargs["original_run_id"] = self._checkpoint.run_id

            logger.info(
                f"Flow {self.id}: resuming from checkpoint {self._checkpoint.id}, "
                f"skipping {len(self._checkpoint.completed_node_ids)} completed nodes"
            )
        else:
            self.reset_run_state()
            self._checkpoint = None

        self._original_input = input_data

        run_id = uuid4()
        wf_run_id = str(config.run_id) if config and config.run_id else str(run_id)
        merged_kwargs = kwargs | {
            "run_id": run_id,
            "parent_run_id": kwargs.get("parent_run_id", run_id),
        }

        if self._should_checkpoint():
            if self._checkpoint:
                self._checkpoint.run_id = str(run_id)
                self._checkpoint.wf_run_id = wf_run_id
                self._checkpoint.status = CheckpointStatus.ACTIVE
                self._save_checkpoint()
            else:
                self._checkpoint = FlowCheckpoint(
                    flow_id=self.id,
                    run_id=str(run_id),
                    wf_run_id=wf_run_id,
                    original_input=input_data,
                    original_config=config.model_dump() if config else None,
                )
                self._save_checkpoint()

        config = self._setup_checkpoint_context(config)

        logger.info(f"Flow {self.id}: execution started.")
        self.run_on_flow_start(input_data, config, **merged_kwargs)
        time_start = datetime.now()

        try:
            if self.nodes:
                while self._ts.is_active():
                    ready_nodes = self._get_nodes_ready_to_run(input_data=input_data)

                    if self._checkpoint:
                        already_completed = [
                            n.node.id for n in ready_nodes if n.node.id in self._checkpoint.completed_node_ids
                        ]
                        if already_completed:
                            self._ts.done(*already_completed)
                        ready_nodes = [n for n in ready_nodes if n.node.id not in self._checkpoint.completed_node_ids]

                    nodes_to_run = [node for node in ready_nodes if node.is_ready]

                    if nodes_to_run:
                        tasks = [
                            node.node.run_async(
                                input_data=node.input_data,
                                depends_result=node.depends_result,
                                config=config,
                                **(merged_kwargs | {"parent_run_id": run_id}),
                            )
                            for node in nodes_to_run
                        ]

                        results_list = await asyncio.gather(*tasks)

                        results = {node.node.id: result for node, result in zip(nodes_to_run, results_list)}

                        self._results.update(results)
                        self._ts.done(*results.keys())

                        if self._should_checkpoint():
                            self._update_checkpoint(results, CheckpointStatus.ACTIVE)

                    # Wait for ready nodes to be processed and reduce CPU usage by yielding control to the event loop
                    await asyncio.sleep(0.003)

            output = self._get_output()
            failed_nodes = self._get_failed_nodes_with_raise_behavior()

            if failed_nodes:
                if self._checkpoint and self._should_checkpoint_on_failure():
                    self._update_checkpoint({}, CheckpointStatus.FAILED)

                failed_names = [node.name or node.id for node in failed_nodes]
                error_msg = f"Flow execution failed due to node failures: {', '.join(failed_names)}"
                error = FlowNodeFailureException(error_msg, failed_nodes)
                self.run_on_flow_error(error, config, failed_nodes=failed_nodes, **merged_kwargs)
                logger.error(f"Flow {self.id}: execution failed in {format_duration(time_start, datetime.now())}.")
                return RunnableResult(
                    status=RunnableStatus.FAILURE,
                    input=input_data,
                    output=output,
                    error=RunnableResultError.from_exception(error, failed_nodes=failed_nodes),
                )

            if self._should_checkpoint():
                self._update_checkpoint({}, CheckpointStatus.COMPLETED)
                self._cleanup_old_checkpoints()

            self.run_on_flow_end(output, config, **merged_kwargs)
            logger.info(f"Flow {self.id}: execution succeeded in {format_duration(time_start, datetime.now())}.")
            return RunnableResult(status=RunnableStatus.SUCCESS, input=input_data, output=output)
        except Exception as e:
            if self._checkpoint and self._should_checkpoint_on_failure():
                self._update_checkpoint({}, CheckpointStatus.FAILED)

            failed_nodes = self._get_failed_nodes_with_raise_behavior()
            self.run_on_flow_error(e, config, failed_nodes=failed_nodes, **merged_kwargs)
            logger.error(f"Flow {self.id}: execution failed in {format_duration(time_start, datetime.now())}.")
            return RunnableResult(
                status=RunnableStatus.FAILURE,
                input=input_data,
                error=RunnableResultError.from_exception(e, failed_nodes=failed_nodes),
            )
        finally:
            try:
                await self._cleanup_dry_run_async(config)
            except Exception as e:
                logger.error(f"Async dry-run cleanup failed: {e}")

    def get_dependant_nodes(
        self, nodes_types_to_skip: set[str] | None = None
    ) -> list[Node]:
        """
        Gets the list of dependent nodes in the flow.

        Args:
            nodes_types_to_skip (set[NodeType] | None, optional): Set of node types to skip. Defaults to None.

        Returns:
            list[Node]: List of dependent nodes.
        """
        if not nodes_types_to_skip:
            nodes_types_to_skip = set()

        return [
            dep.node
            for node in self.nodes
            if node.type not in nodes_types_to_skip
            for dep in node.depends
        ]

    def get_non_dependant_nodes(
        self, nodes_types_to_skip: set[str] | None = None
    ) -> list[Node]:
        """
        Gets the list of non-dependent nodes in the flow.

        Args:
            nodes_types_to_skip (set[NodeType] | None, optional): Set of node types to skip. Defaults to None.

        Returns:
            list[Node]: List of non-dependent nodes.
        """
        if not nodes_types_to_skip:
            nodes_types_to_skip = set()

        dependant_nodes = self.get_dependant_nodes(
            nodes_types_to_skip=nodes_types_to_skip
        )
        return [
            node
            for node in self.nodes
            if node.type not in nodes_types_to_skip and node not in dependant_nodes
        ]

    def _load_checkpoint(self, resume_from: str | FlowCheckpoint) -> FlowCheckpoint | None:
        """Load checkpoint from ID or return if already a FlowCheckpoint instance."""
        if isinstance(resume_from, FlowCheckpoint):
            return resume_from

        cfg = self._effective_checkpoint_config or self.checkpoint
        if cfg.backend:
            return cfg.backend.load(resume_from)

        return None

    def _restore_from_checkpoint(self, checkpoint: FlowCheckpoint) -> None:
        """Restore flow state from checkpoint.

        Args:
            checkpoint: The checkpoint to restore from
        """
        self._results = {}

        for node_id, node_state in checkpoint.node_states.items():
            if node_state.status in ("success", "failure", "skip"):
                error = None
                if node_state.error:
                    error_data = dict(node_state.error)
                    error_type = error_data.get("type")
                    if isinstance(error_type, str):
                        error_data["type"] = getattr(builtins, error_type, Exception)
                    error = RunnableResultError(**error_data)

                self._results[node_id] = RunnableResult(
                    status=RunnableStatus(node_state.status),
                    input=node_state.input_data,
                    output=node_state.output_data,
                    error=error,
                )

            node = self._node_by_id.get(node_id)
            if node and isinstance(node, CheckpointMixin) and node_state.internal_state:
                node.from_checkpoint_state(node_state.internal_state)
                logger.debug(f"Flow {self.id}: restored internal state for node {node_id}")

        self._ts = self.init_node_topological_sorter(nodes=self.nodes)

        if checkpoint.has_pending_inputs():
            pending_node_ids = list(checkpoint.pending_inputs.keys())
            logger.info(
                f"Flow {self.id}: checkpoint has {len(pending_node_ids)} nodes waiting for input: {pending_node_ids}. "
                f"These nodes will re-request approval on resume."
            )
            for node_id in pending_node_ids:
                checkpoint.clear_pending_input(node_id)

        logger.info(
            f"Flow {self.id}: restored from checkpoint - "
            f"{len(checkpoint.completed_node_ids)} nodes completed, "
            f"{len(checkpoint.pending_node_ids)} nodes pending"
        )

    def _setup_checkpoint_context(self, config: RunnableConfig | None) -> RunnableConfig | None:
        """Setup checkpoint context for HITL and mid-agent-loop checkpointing."""
        if not self._should_checkpoint():
            return config

        def on_pending_input(node_id: str, prompt: str, metadata: dict | None) -> None:
            if self._checkpoint:
                self._checkpoint.mark_pending_input(node_id, prompt, metadata)
                self._save_checkpoint()
                logger.info(f"Flow {self.id}: checkpoint saved with PENDING_INPUT status for node {node_id}")

        def on_input_received(node_id: str) -> None:
            if self._checkpoint:
                self._checkpoint.clear_pending_input(node_id)
                self._save_checkpoint()
                logger.debug(f"Flow {self.id}: cleared pending input for node {node_id}")

        def on_save_mid_run(node_id: str) -> None:
            cfg = self._effective_checkpoint_config or self.checkpoint
            if self._checkpoint and cfg.checkpoint_mid_agent_loop:
                node = self._node_by_id.get(node_id)
                if node and isinstance(node, CheckpointMixin):
                    checkpoint_state = node.to_checkpoint_state()
                    internal_state = (
                        checkpoint_state.model_dump() if hasattr(checkpoint_state, "model_dump") else checkpoint_state
                    )
                    if node_id in self._checkpoint.node_states:
                        self._checkpoint.node_states[node_id].internal_state = internal_state
                    else:
                        self._checkpoint.node_states[node_id] = NodeCheckpointState(
                            node_id=node_id,
                            node_type=node.type,
                            status="active",
                            internal_state=internal_state,
                        )
                self._save_checkpoint()
                logger.debug(f"Flow {self.id}: mid-run checkpoint saved for node {node_id}")

        checkpoint_context = CheckpointContext(
            on_pending_input=on_pending_input,
            on_input_received=on_input_received,
            on_save_mid_run=on_save_mid_run,
        )

        if config is None:
            config = RunnableConfig(checkpoint=CheckpointConfig(context=checkpoint_context))
        elif config.checkpoint:
            config.checkpoint.context = checkpoint_context
        else:
            config.checkpoint = CheckpointConfig(context=checkpoint_context)

        return config

    def _get_effective_checkpoint_config(self, config: RunnableConfig | None) -> CheckpointConfig:
        """Merge flow-level checkpoint defaults with per-run overrides from RunnableConfig.

        Run-level values override flow-level defaults. Fields not explicitly set at run-level
        (i.e. left at their default) fall back to the flow-level value.
        """
        base = self.checkpoint
        if not config or not config.checkpoint:
            return base

        run_cfg = config.checkpoint
        run_fields = run_cfg.model_fields_set

        merged_values: dict = {}
        for field_name in CheckpointConfig.model_fields:
            if field_name in run_fields:
                merged_values[field_name] = getattr(run_cfg, field_name)
            else:
                merged_values[field_name] = getattr(base, field_name)

        return CheckpointConfig(**merged_values)

    def _should_checkpoint(self) -> bool:
        """Check if checkpointing is enabled and configured."""
        cfg = self._effective_checkpoint_config or self.checkpoint
        return cfg.enabled and cfg.backend is not None and cfg.checkpoint_after_node

    def _should_checkpoint_on_failure(self) -> bool:
        """Check if should checkpoint on failure."""
        cfg = self._effective_checkpoint_config or self.checkpoint
        return cfg.enabled and cfg.backend is not None and cfg.checkpoint_on_failure

    def _update_checkpoint(self, new_results: dict[str, RunnableResult], status: CheckpointStatus) -> None:
        """Update checkpoint with new node results."""
        if not self._checkpoint:
            return

        for node_id, result in new_results.items():
            node = self._node_by_id.get(node_id)
            if not node:
                continue

            cfg = self._effective_checkpoint_config or self.checkpoint
            if node_id in cfg.exclude_node_ids:
                continue

            internal_state = {}
            if isinstance(node, CheckpointMixin):
                checkpoint_state = node.to_checkpoint_state()
                internal_state = (
                    checkpoint_state.model_dump() if hasattr(checkpoint_state, "model_dump") else checkpoint_state
                )

            node_state = NodeCheckpointState(
                node_id=node_id,
                node_type=node.type,
                status=result.status.value,
                input_data=result.input,
                output_data=result.output,
                error=result.error.to_dict() if result.error else None,
                internal_state=internal_state,
                completed_at=datetime.now(),
            )

            self._checkpoint.mark_node_complete(node_id, node_state)

        self._checkpoint.status = status
        self._save_checkpoint()

    def _save_checkpoint(self) -> None:
        """Save current checkpoint to backend."""
        cfg = self._effective_checkpoint_config or self.checkpoint
        if self._checkpoint and cfg.backend:
            try:
                cfg.backend.save(self._checkpoint)
                logger.debug(f"Flow {self.id}: checkpoint saved - {self._checkpoint.id}")
            except Exception as e:
                logger.warning(f"Flow {self.id}: failed to save checkpoint - {e}")

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints beyond max_checkpoints."""
        cfg = self._effective_checkpoint_config or self.checkpoint
        if not cfg.backend:
            return

        try:
            deleted = cfg.backend.cleanup_by_flow(
                self.id,
                keep_count=cfg.max_checkpoints,
                older_than_hours=cfg.max_retention_hours,
            )
            if deleted > 0:
                logger.debug(f"Flow {self.id}: cleaned up {deleted} old checkpoints")
        except Exception as e:
            logger.warning(f"Flow {self.id}: failed to cleanup checkpoints - {e}")

    def list_checkpoints(self, limit: int = 10) -> list[FlowCheckpoint]:
        """
        List checkpoints for this flow.

        Args:
            limit: Maximum number of checkpoints to return

        Returns:
            List of checkpoints, newest first

        Example:
            >>> checkpoints = flow.list_checkpoints(limit=5)
            >>> for cp in checkpoints:
            ...     print(f"{cp.id}: {cp.status}")
        """
        if not self.checkpoint.backend:
            return []
        return self.checkpoint.backend.get_list_by_flow(self.id, limit=limit)

    def get_latest_checkpoint(self, status: CheckpointStatus | None = None) -> FlowCheckpoint | None:
        """
        Get the most recent checkpoint.

        Args:
            status: Optional status filter

        Returns:
            The latest checkpoint if found

        Example:
            >>> # Get latest successful checkpoint
            >>> cp = flow.get_latest_checkpoint(status=CheckpointStatus.COMPLETED)
        """
        if not self.checkpoint.backend:
            return None
        return self.checkpoint.backend.get_latest_by_flow(self.id, status=status)

    def get_pending_inputs(self, checkpoint_id: str | None = None) -> dict[str, PendingInputContext]:
        """
        Get pending input contexts (HITL) from a checkpoint.

        Useful for UI to display what approvals are waiting.

        Args:
            checkpoint_id: Specific checkpoint ID, or None to use current/latest

        Returns:
            Dict of node_id -> PendingInputContext for nodes waiting for input

        Example:
            >>> pending = flow.get_pending_inputs()
            >>> for node_id, ctx in pending.items():
            ...     print(f"Node {node_id} asks: {ctx.prompt}")
        """
        checkpoint = None
        if checkpoint_id:
            checkpoint = self.checkpoint.backend.load(checkpoint_id) if self.checkpoint.backend else None
        elif self._checkpoint:
            checkpoint = self._checkpoint
        elif self.checkpoint.backend:
            checkpoint = self.checkpoint.backend.get_latest_by_flow(self.id)

        if checkpoint:
            return checkpoint.pending_inputs
        return {}

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Delete a specific checkpoint.

        Args:
            checkpoint_id: The checkpoint to delete

        Returns:
            True if deleted, False if not found
        """
        if not self.checkpoint.backend:
            return False
        return self.checkpoint.backend.delete(checkpoint_id)

    def clear_all_checkpoints(self) -> int:
        """
        Delete all checkpoints for this flow.

        Returns:
            Number of checkpoints deleted
        """
        if not self.checkpoint.backend:
            return 0
        return self.checkpoint.backend.cleanup_by_flow(self.id, keep_count=0)

    def add_nodes(self, nodes: Node | list[Node]):
        """
        Add one or more nodes to the flow.

        Args:
            nodes (Node or list[Node]): Node(s) to add to the flow.

        Raises:
            TypeError: If 'nodes' is not a Node or a list of Node.
            ValueError: If 'nodes' is an empty list, if a node with the same id already exists in the flow,
                        or if there are duplicate node ids in the input list.
        """

        if nodes is None:
            raise ValueError("No node provided. Nodes cannot be None.")

        # Convert a single Node to a list for consistent handling
        if isinstance(nodes, Node):
            nodes = [nodes]

        # Check if it's a valid list of nodes
        if not isinstance(nodes, list) or not all(isinstance(n, Node) for n in nodes):
            raise TypeError("Nodes must be a Node instance or a list of Node instances.")

        if not nodes:
            raise ValueError("Cannot add an empty list of nodes to the flow.")

        # Add nodes to the flow, checking for duplicates in the flow
        for node in nodes:
            if node.id in self._node_by_id:
                raise ValueError(f"Node with id {node.id} already exists in the flow.")

            self.nodes.append(node)
            self._node_by_id[node.id] = node
            if node.is_postponed_component_init:
                node.init_components(self.connection_manager)
        self.reset_run_state()

        return self  # enable chaining
