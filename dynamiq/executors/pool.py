import os
from concurrent import futures

import jsonpickle

from dynamiq.executors.base import BaseExecutor
from dynamiq.nodes.node import Node, NodeReadyToRun
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus
from dynamiq.utils.logger import logger

MAX_WORKERS_THREAD_POOL_EXECUTOR = 8
MAX_WORKERS_PROCESS_POOL_EXECUTOR = os.cpu_count()


class PoolExecutor(BaseExecutor):
    """
    A pool executor that manages concurrent execution of nodes using either ThreadPoolExecutor or
    ProcessPoolExecutor.

    Args:
        pool_executor (type): The type of pool executor to use (ThreadPoolExecutor or
            ProcessPoolExecutor).
        max_workers (int, optional): The maximum number of workers in the pool. Defaults to None.
    """

    def __init__(
        self,
        pool_executor: (
            type[futures.ThreadPoolExecutor] | type[futures.ProcessPoolExecutor]
        ),
        max_workers: int | None = None,
    ):
        super().__init__(max_workers=max_workers)
        self.executor = pool_executor(max_workers=max_workers)
        self.node_by_future = {}

    def shutdown(self, wait: bool = True):
        """
        Shuts down the executor.

        Args:
            wait (bool, optional): Whether to wait for pending futures to complete. Defaults to True.
        """
        self.executor.shutdown(wait=wait)

    def execute(
        self,
        ready_nodes: list[NodeReadyToRun],
        config: RunnableConfig = None,
        **kwargs,
    ) -> dict[str, RunnableResult]:
        """
        Executes the given ready nodes and returns their results.

        Args:
            ready_nodes (list[NodeReadyToRun]): List of nodes ready to run.
            config (RunnableConfig, optional): Configuration for the execution. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            dict[str, RunnableResult]: A dictionary of node IDs and their execution results.
        """
        self.run_nodes(ready_nodes=ready_nodes, config=config, **kwargs)
        completed_node_futures, _ = futures.wait(
            fs=self.node_by_future.keys(), return_when=futures.FIRST_COMPLETED
        )
        results = self.complete_nodes(completed_node_futures=completed_node_futures)

        return results

    def run_nodes(
        self,
        ready_nodes: list[NodeReadyToRun],
        config: RunnableConfig = None,
        **kwargs,
    ):
        """
        Submits ready nodes for execution.

        Args:
            ready_nodes (list[NodeReadyToRun]): List of nodes ready to run.
            config (RunnableConfig, optional): Configuration for the execution. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        for ready_node in ready_nodes:
            if ready_node.is_ready:
                future = self.run_node(ready_node=ready_node, config=config, **kwargs)
                self.node_by_future[future] = ready_node.node
            else:
                logger.error(
                    f"Node {ready_node.node.name} - {ready_node.node.id}: not ready to run."
                )

    def complete_nodes(
        self, completed_node_futures: list[futures.Future]
    ) -> dict[str, RunnableResult]:
        """
        Processes completed node futures and returns their results.

        Args:
            completed_node_futures (list[futures.Future]): List of completed node futures.

        Returns:
            dict[str, RunnableResult]: A dictionary of node IDs and their execution results.
        """
        results = {}
        for f in completed_node_futures:
            node = self.node_by_future.pop(f)
            try:
                node_result: RunnableResult = f.result()
            except Exception as e:
                logger.error(
                    f"Node {node.name} - {node.id}: execution failed due the unexpected error. Error: {e}"
                )
                node_result = RunnableResult(status=RunnableStatus.FAILURE)

            results[node.id] = node_result

        return results

    def run_node(self, ready_node: NodeReadyToRun, config: RunnableConfig = None, **kwargs):
        """
        Submits ready node for execution.

        Args:
            ready_node (NodeReadyToRun): node ready to run.
            config (RunnableConfig, optional): Configuration for the execution. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        return self.executor.submit(
            ready_node.node.run,
            input_data=ready_node.input_data,
            config=config,
            depends_result=ready_node.depends_result,
            **kwargs,
        )


class ThreadExecutor(PoolExecutor):
    """
    A thread-based pool executor.

    Args:
        max_workers (int, optional): The maximum number of worker threads. Defaults to None.
    """

    def __init__(self, max_workers: int | None = None):
        max_workers = max_workers or MAX_WORKERS_THREAD_POOL_EXECUTOR
        super().__init__(
            pool_executor=futures.ThreadPoolExecutor, max_workers=max_workers
        )


class ProcessExecutor(PoolExecutor):
    """
    A process-based pool executor.

    Args:
        max_workers (int, optional): The maximum number of worker processes. Defaults to None.
    """

    def __init__(self, max_workers: int | None = None):
        max_workers = max_workers or MAX_WORKERS_PROCESS_POOL_EXECUTOR
        super().__init__(
            pool_executor=futures.ProcessPoolExecutor, max_workers=max_workers
        )

    @staticmethod
    def serialize_node(node: Node) -> str:
        """
        Serializes node data.

        Args:
            node (Node): Node instance.

        Returns:
            str: Serialized node data
        """
        return jsonpickle.encode(node)

    @staticmethod
    def deserialize_node(node_data: str) -> Node:
        """
        Deserializes node data.

        Args:
            node_data (str): Serialized node data.

        Returns:
            Node: Node instance.
        """
        return jsonpickle.decode(node_data)  # nosec

    @classmethod
    def _run_node(cls, node_data: str, **kwargs) -> RunnableResult:
        """
        Deserializes node data and runs the node.

        Args:
            node_data (str): Serialized node data.
            **kwargs: Additional keyword arguments.
        """
        node_instance = cls.deserialize_node(node_data)
        return node_instance.run(**kwargs)

    def run_node(self, ready_node: NodeReadyToRun, config: RunnableConfig = None, **kwargs):
        """
        Submits ready node for execution.

        Args:
            ready_node (NodeReadyToRun): node ready to run.
            config (RunnableConfig, optional): Configuration for the execution. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        return self.executor.submit(
            self._run_node,
            node_data=self.serialize_node(ready_node.node),
            input_data=ready_node.input_data,
            config=config,
            depends_result=ready_node.depends_result,
            **kwargs,
        )
