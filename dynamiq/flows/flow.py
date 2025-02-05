from datetime import datetime
from graphlib import CycleError, TopologicalSorter
from io import BytesIO
from typing import Any
from uuid import uuid4

from pydantic import Field, field_validator

from dynamiq.connections.managers import ConnectionManager
from dynamiq.executors.base import BaseExecutor
from dynamiq.executors.pool import ThreadExecutor
from dynamiq.flows.base import BaseFlow
from dynamiq.nodes.node import Node, NodeReadyToRun
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus
from dynamiq.utils.duration import format_duration
from dynamiq.utils.logger import logger


class Flow(BaseFlow):
    """
    A class for managing and executing a graph-like structure of nodes.

    Attributes:
        nodes (list[Node]): List of nodes in the flow.
        executor (type[BaseExecutor]): Executor class for running nodes. Defaults to ThreadExecutor.
        max_node_workers (int | None): Maximum number of concurrent node workers. Defaults to None.
        connection_manager (ConnectionManager): Manager for handling connections. Defaults to ConnectionManager().
    """

    nodes: list[Node] = []
    executor: type[BaseExecutor] = ThreadExecutor
    max_node_workers: int | None = None
    connection_manager: ConnectionManager = Field(default_factory=ConnectionManager)

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

    @property
    def to_dict_exclude_params(self):
        return {"nodes": True, "connection_manager": True}

    def to_dict(self, include_secure_params: bool = True, **kwargs) -> dict:
        """Converts the instance to a dictionary.

        Returns:
            dict: A dictionary representation of the instance.
        """
        data = super().to_dict(include_secure_params=include_secure_params, **kwargs)
        data["nodes"] = [node.to_dict(include_secure_params=include_secure_params, **kwargs) for node in self.nodes]
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
        for node_id in ready_ts_nodes:
            node = self._node_by_id[node_id]
            depends_result = {}
            is_ready = True
            for dep in node.depends:
                if (
                    dep_result := self._results.get(dep.node.id)
                ) and dep_result.status != RunnableStatus.UNDEFINED:
                    depends_result[dep.node.id] = dep_result
                else:
                    is_ready = False

            ready_node = NodeReadyToRun(
                node=node,
                is_ready=is_ready,
                input_data=input_data,
                depends_result=depends_result,
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

    def run(self, input_data: Any, config: RunnableConfig = None, **kwargs):
        """
        Runs the flow with the given input data and configuration.

        Args:
            input_data (Any): Input data for the flow.
            config (RunnableConfig, optional): Configuration for the run. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            RunnableResult: Result of the flow execution.
        """
        self.reset_run_state()
        run_id = uuid4()
        merged_kwargs = kwargs | {
            "run_id": run_id,
            "parent_run_id": kwargs.get("parent_run_id", run_id),
        }

        logger.info(f"Flow {self.id}: execution started.")
        self.run_on_flow_start(input_data, config, **merged_kwargs)
        time_start = datetime.now()

        try:
            if self.nodes:
                max_workers = (
                    config.max_node_workers if config else self.max_node_workers
                )
                run_executor = self.executor(max_workers=max_workers)

                while self._ts.is_active():
                    ready_nodes = self._get_nodes_ready_to_run(input_data=input_data)
                    results = run_executor.execute(
                        ready_nodes=ready_nodes,
                        config=config,
                        **(merged_kwargs | {"parent_run_id": run_id}),
                    )
                    self._results.update(results)
                    self._ts.done(*results.keys())

                run_executor.shutdown()

            output = self._get_output()
            self.run_on_flow_end(self._get_output(), config, **merged_kwargs)
            logger.info(
                f"Flow {self.id}: execution succeeded in {format_duration(time_start, datetime.now())}."
            )
            return RunnableResult(
                status=RunnableStatus.SUCCESS, input=input_data, output=output
            )
        except Exception as e:
            self.run_on_flow_error(e, config, **merged_kwargs)
            logger.error(
                f"Flow {self.id}: execution failed in "
                f"{format_duration(time_start, datetime.now())}."
            )
            return RunnableResult(
                status=RunnableStatus.FAILURE,
                input=input_data,
            )

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
