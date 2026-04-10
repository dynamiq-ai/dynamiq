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

from dynamiq.checkpoints.checkpoint import CheckpointFlowMixin, CheckpointNodeMixin, CheckpointStatus, FlowCheckpoint
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


class Flow(CheckpointFlowMixin, BaseFlow):
    """
    A class for managing and executing a graph-like structure of nodes.

    Attributes:
        nodes (list[Node]): List of nodes in the flow.
        executor (type[BaseExecutor]): Executor class for running nodes. Defaults to ThreadExecutor.
        max_node_workers (int | None): Maximum number of concurrent node workers. Defaults to None.
        connection_manager (ConnectionManager): Manager for handling connections. Defaults to ConnectionManager().
        checkpoint (CheckpointConfig): Flow-level checkpoint defaults (backend, retention, behavior).
            Inherited from CheckpointFlowMixin.

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
        data["checkpoint"] = self.checkpoint.to_dict()
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

    def _restore_from_checkpoint(self, checkpoint: FlowCheckpoint) -> None:
        """Restore flow execution state from a persisted checkpoint."""
        self._results = {}

        for node_id, node_state in checkpoint.node_states.items():
            if node_state.status in (
                RunnableStatus.SUCCESS.value,
                RunnableStatus.FAILURE.value,
                RunnableStatus.SKIP.value,
            ):
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
            if node and isinstance(node, CheckpointNodeMixin) and node_state.internal_state:
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

    def reset_run_state(self):
        """Resets the run state of the flow and clears stale resumed flags on all nodes."""
        self._results = {
            node.id: RunnableResult(status=RunnableStatus.UNDEFINED)
            for node in self.nodes
        }
        self._ts = self.init_node_topological_sorter(nodes=self.nodes)
        self._checkpoint_persisted = False
        for node in self.nodes:
            if isinstance(node, CheckpointNodeMixin) and node.is_resumed:
                node.reset_resumed_flag()

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

            # The checkpoint already exists in the backend; mark it persisted so
            # the first _save_checkpoint() call creates a new APPEND snapshot
            # instead of overwriting the loaded one in-place.
            self._checkpoint_persisted = True
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

        if self._is_checkpoint_active():
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
                    original_config=config.to_checkpoint_dict() if config else None,
                    pending_node_ids=[n.id for n in self.nodes],
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
                node_run_kwargs = merged_kwargs | {"parent_run_id": run_id}

                while self._ts.is_active():
                    ready_nodes = self._get_nodes_ready_to_run(input_data=input_data)

                    if self._checkpoint:
                        already_completed = [
                            n.node.id for n in ready_nodes if n.node.id in self._checkpoint.completed_node_ids
                        ]
                        if already_completed:
                            self._ts.done(*already_completed)
                        ready_nodes = [n for n in ready_nodes if n.node.id not in self._checkpoint.completed_node_ids]

                    results = run_executor.execute(
                        ready_nodes=ready_nodes,
                        config=config,
                        **node_run_kwargs,
                    )
                    self._results.update(results)
                    self._ts.done(*results.keys())

                    if self._is_checkpoint_after_node_enabled():
                        self._update_checkpoint(results, CheckpointStatus.ACTIVE)

                    time.sleep(0.001)

                run_executor.shutdown()

            output = self._get_output()
            failed_nodes = self._get_failed_nodes_with_raise_behavior()

            if failed_nodes:
                if self._checkpoint and self._is_checkpoint_on_failure_enabled():
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

            if self._is_checkpoint_after_node_enabled():
                self._update_checkpoint({}, CheckpointStatus.COMPLETED)
                self._cleanup_old_checkpoints()

            self.run_on_flow_end(output, config, **merged_kwargs)
            logger.info(f"Flow {self.id}: execution succeeded in {format_duration(time_start, datetime.now())}.")
            return RunnableResult(status=RunnableStatus.SUCCESS, input=input_data, output=output)
        except Exception as e:
            if self._checkpoint and self._is_checkpoint_on_failure_enabled():
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
            self._checkpoint = await self._load_checkpoint_async(effective_resume)
            if not self._checkpoint:
                raise ValueError(f"Checkpoint not found: {effective_resume}")

            # The checkpoint already exists in the backend; mark it persisted so
            # the first _save_checkpoint_async() call creates a new APPEND snapshot
            # instead of overwriting the loaded one in-place.
            self._checkpoint_persisted = True
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

        if self._is_checkpoint_active():
            if self._checkpoint:
                self._checkpoint.run_id = str(run_id)
                self._checkpoint.wf_run_id = wf_run_id
                self._checkpoint.status = CheckpointStatus.ACTIVE
                await self._save_checkpoint_async()
            else:
                self._checkpoint = FlowCheckpoint(
                    flow_id=self.id,
                    run_id=str(run_id),
                    wf_run_id=wf_run_id,
                    original_input=input_data,
                    original_config=config.to_checkpoint_dict() if config else None,
                    pending_node_ids=[n.id for n in self.nodes],
                )
                await self._save_checkpoint_async()

        config = self._setup_checkpoint_context(config)

        logger.info(f"Flow {self.id}: execution started.")
        self.run_on_flow_start(input_data, config, **merged_kwargs)
        time_start = datetime.now()

        try:
            if self.nodes:
                node_run_kwargs = merged_kwargs | {"parent_run_id": run_id}

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
                                **node_run_kwargs,
                            )
                            for node in nodes_to_run
                        ]

                        results_list = await asyncio.gather(*tasks)

                        results = {node.node.id: result for node, result in zip(nodes_to_run, results_list)}

                        self._results.update(results)
                        self._ts.done(*results.keys())

                        if self._is_checkpoint_after_node_enabled():
                            await self._update_checkpoint_async(results, CheckpointStatus.ACTIVE)

                    # Wait for ready nodes to be processed and reduce CPU usage by yielding control to the event loop
                    await asyncio.sleep(0.001)

            output = self._get_output()
            failed_nodes = self._get_failed_nodes_with_raise_behavior()

            if failed_nodes:
                if self._checkpoint and self._is_checkpoint_on_failure_enabled():
                    await self._update_checkpoint_async({}, CheckpointStatus.FAILED)
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

            if self._is_checkpoint_after_node_enabled():
                await self._update_checkpoint_async({}, CheckpointStatus.COMPLETED)
                await self._cleanup_old_checkpoints_async()

            self.run_on_flow_end(output, config, **merged_kwargs)
            logger.info(f"Flow {self.id}: execution succeeded in {format_duration(time_start, datetime.now())}.")
            return RunnableResult(status=RunnableStatus.SUCCESS, input=input_data, output=output)
        except Exception as e:
            if self._checkpoint and self._is_checkpoint_on_failure_enabled():
                await self._update_checkpoint_async({}, CheckpointStatus.FAILED)
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
