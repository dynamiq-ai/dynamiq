from abc import ABC, abstractmethod

from dynamiq.nodes.node import NodeReadyToRun
from dynamiq.runnables import RunnableConfig, RunnableResult


class BaseExecutor(ABC):
    """
    Abstract base class for executors that run nodes in a workflow.

    Attributes:
        max_workers (int | None): Maximum number of concurrent workers. None means no limit.
    """

    def __init__(self, max_workers: int | None = None):
        """
        Initialize the BaseExecutor.

        Args:
            max_workers (int | None, optional): Maximum number of concurrent workers. Defaults to None.
        """
        self.max_workers = max_workers

    @abstractmethod
    def shutdown(self, wait: bool = True):
        """
        Shut down the executor.

        Args:
            wait (bool, optional): Whether to wait for pending tasks to complete. Defaults to True.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def execute(
        self,
        ready_nodes: list[NodeReadyToRun],
        config: RunnableConfig = None,
        **kwargs,
    ) -> dict[str, RunnableResult]:
        """
        Execute the given nodes that are ready to run.

        Args:
            ready_nodes (list[NodeReadyToRun]): List of nodes ready for execution.
            config (RunnableConfig, optional): Configuration for the execution. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            dict[str, RunnableResult]: A dictionary mapping node IDs to their execution results.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError
