from abc import ABC, abstractmethod
from enum import Enum
from io import BytesIO
from typing import Any, Awaitable

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.cache.config import CacheConfig
from dynamiq.callbacks import BaseCallbackHandler
from dynamiq.types.streaming import StreamingConfig
from dynamiq.utils import format_value, generate_uuid, is_called_from_async_context


class NodeRunnableConfig(BaseModel):
    streaming: StreamingConfig | None = None


class RunnableConfig(BaseModel):
    """
    Configuration class for Runnable objects.

    Attributes:
        callbacks (list[BaseCallbackHandler]): List of callback handlers.
        cache (CacheConfig | None): Cache configuration.
        max_node_workers (int | None): Maximum number of node workers.
    """

    run_id: str | None = Field(default_factory=generate_uuid)
    callbacks: list[BaseCallbackHandler] = []
    cache: CacheConfig | None = None
    max_node_workers: int | None = None
    nodes_override: dict[str, NodeRunnableConfig] = {}

    model_config = ConfigDict(arbitrary_types_allowed=True)


class RunnableStatus(str, Enum):
    """
    Enumeration of possible statuses for a Runnable object.

    Attributes:
        UNDEFINED: Undefined status.
        FAILURE: Failure status.
        SUCCESS: Success status.
        SKIP: Skip status.
    """

    UNDEFINED = "undefined"
    FAILURE = "failure"
    SUCCESS = "success"
    SKIP = "skip"


class RunnableResult(BaseModel):
    """
    Dataclass representing the result of a Runnable execution.

    Attributes:
        status (RunnableStatus): The status of the execution.
        input (Any): The input data of the execution.
        output (Any): The output data of the execution.
    """

    status: RunnableStatus
    input: Any = None
    output: Any = None

    def to_depend_dict(
        self,
        skip_format_types: set | None = None,
        force_format_types: set | None = None,
        **kwargs
    ) -> dict:
        """
        Convert the RunnableResult instance to a dictionary that used as dependency context.

        Returns:
            dict: A dictionary representation of the RunnableResult.
        """
        if skip_format_types is None:
            skip_format_types = set()
        skip_format_types.update({BytesIO, BaseModel, bytes})

        if force_format_types is None:
            force_format_types = set()
        force_format_types.add(RunnableResult)

        return self.to_dict(skip_format_types, force_format_types, **kwargs)

    def to_tracing_depend_dict(
        self, skip_format_types: set | None = None, force_format_types: set | None = None, **kwargs
    ) -> dict:
        """
        Convert the RunnableResult instance to a dictionary that used as dependency context in tracing.

        Returns:
            dict: A dictionary representation of the RunnableResult.
        """

        depend_dict = self.to_depend_dict(skip_format_types, force_format_types, **kwargs)
        depend_dict.pop("input", None)

        return depend_dict

    def to_dict(
        self,
        skip_format_types: set | None = None,
        force_format_types: set | None = None,
        **kwargs
    ) -> dict:
        """
        Convert the RunnableResult instance to a dictionary.

        Returns:
            dict: A dictionary representation of the RunnableResult.
        """

        return {
            "status": self.status.value,
            "input": format_value(self.input, skip_format_types, force_format_types)[0],
            "output": format_value(self.output, skip_format_types, force_format_types)[0],
        }


class Runnable(ABC):
    """
    Abstract base class for runnable objects.
    """

    def run(
        self, input_data: Any, config: RunnableConfig = None, is_async: bool | None = None, **kwargs
    ) -> RunnableResult | Awaitable[RunnableResult]:
        """Run the workflow with given input data and configuration.

        This method acts as a dispatcher based on whether it's called from an async context:
        - In synchronous contexts, it calls run_sync
        - In asynchronous contexts, it calls run_async when awaited
        - The mode can be explicitly specified with the is_async parameter

        For direct control, use run_sync() or run_async() methods.

        Args:
            input_data (Any): Input data for the workflow.
            config (RunnableConfig, optional): Configuration for the run. Defaults to None.
            is_async (bool, optional): Force synchronous or asynchronous execution.
                If None, tries to detect based on calling context.
            **kwargs: Additional keyword arguments.

        Returns:
            Union[RunnableResult, Awaitable[RunnableResult]]: Result of the workflow execution
                or awaitable coroutine leading to the result.
        """
        if is_async is None:
            is_async = is_called_from_async_context()

        if is_async:
            return self.run_async(input_data, config, **kwargs)
        else:
            return self.run_sync(input_data, config, **kwargs)

    @abstractmethod
    def run_sync(self, input_data: Any, config: RunnableConfig = None, **kwargs) -> RunnableResult:
        """
        Abstract method to run the Runnable object synchronously.

        Args:
            input_data (Any): The input data for the execution.
            config (RunnableConfig, optional): Configuration for the execution.
            **kwargs: Additional keyword arguments.

        Returns:
            RunnableResult: The result of the execution.
        """
        pass

    @abstractmethod
    async def run_async(self, input_data: Any, config: RunnableConfig = None, **kwargs) -> RunnableResult:
        """
        Abstract method to run the Runnable object asynchronously.

        Args:
            input_data (Any): The input data for the execution.
            config (RunnableConfig, optional): Configuration for the execution.
            **kwargs: Additional keyword arguments.

        Returns:
            RunnableResult: The result of the execution.
        """
        pass
