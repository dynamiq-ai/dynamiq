import contextvars
import functools
import inspect
from abc import ABC
from contextlib import contextmanager
from typing import Any, Callable, Iterator
from uuid import UUID

READS_SERIALIZATION_ONLY = "_reads_serialization_only"

# Serializations shared by marked hooks inside `shared_serialization_scope`, keyed by (id(entity), for_tracing). A
# context variable, so concurrent runs on other tasks or threads never share one.
_shared_serializations: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "dynamiq_shared_serializations", default=None
)


def reads_serialization_only(method: Callable) -> Callable:
    """Mark a callback hook that neither edits nor keeps its `serialized` argument.

    Inside `shared_serialization_scope` every marked hook receives the same serialization of a node, flow or workflow,
    taken the first time one of them needs it, instead of a new `to_dict()` per call. A subclass that overrides a marked
    hook receives new serializations again unless it marks its override too.

    Args:
        method (Callable): The hook.

    Returns:
        Callable: The same hook, marked.
    """
    setattr(method, READS_SERIALIZATION_ONLY, True)
    return method


@contextmanager
def shared_serialization_scope() -> Iterator[None]:
    """Share serializations among marked hooks until the block ends. Inside an open scope, that scope stays in use."""
    if _shared_serializations.get() is not None:
        yield
        return
    token = _shared_serializations.set({})
    try:
        yield
    finally:
        _shared_serializations.reset(token)


def in_shared_serialization_scope(method: Callable) -> Callable:
    """Run `method`, sync or async, inside `shared_serialization_scope`."""
    if inspect.iscoroutinefunction(method):

        @functools.wraps(method)
        async def run_async(*args: Any, **kwargs: Any) -> Any:
            with shared_serialization_scope():
                return await method(*args, **kwargs)

        return run_async

    @functools.wraps(method)
    def run_sync(*args: Any, **kwargs: Any) -> Any:
        with shared_serialization_scope():
            return method(*args, **kwargs)

    return run_sync


def is_noop_hook(callback: Any, name: str) -> bool:
    """Whether the callback's `name` hook is the base handlers' no-op, which needs no call and no serialization."""
    base = getattr(BaseCallbackHandler, name, None)
    return base is not None and getattr(getattr(callback, name, None), "__func__", None) is base


def serialization_for(entity: Any, hook: Callable, for_tracing: bool) -> dict:
    """The `serialized` argument for `hook`: the shared one for a marked hook inside a scope, otherwise a new one."""
    dict_kwargs = {"for_tracing": True} if for_tracing else {}
    shared = _shared_serializations.get()
    if shared is None or not getattr(hook, READS_SERIALIZATION_ONLY, False):
        return entity.to_dict(**dict_kwargs)
    key = (id(entity), for_tracing)
    if (entry := shared.get(key)) is None:
        # The entry holds the entity too, so its id cannot be reused by another object while the scope lives.
        entry = shared[key] = (entity, entity.to_dict(**dict_kwargs))
    return entry[1]


class NodeCallbackHandler(ABC):
    """Abstract class for node callback handlers."""

    def on_node_start(self, serialized: dict[str, Any], input_data: dict[str, Any], **kwargs: Any):
        """Called when the node starts.

        Args:
            serialized (dict[str, Any]): Serialized node data.
            input_data (dict[str, Any]): Input data for the node.
            **kwargs (Any): Additional arguments.
        """
        pass

    def on_node_end(self, serialized: dict[str, Any], output_data: dict[str, Any], **kwargs: Any):
        """Called when the node ends.

        Args:
            serialized (dict[str, Any]): Serialized node data.
            output_data (dict[str, Any]): Output data from the node.
            **kwargs (Any): Additional arguments.
        """
        pass

    def on_node_error(self, serialized: dict[str, Any], error: BaseException, **kwargs: Any):
        """Called when the node errors.

        Args:
            serialized (dict[str, Any]): Serialized node data.
            error (BaseException): Error encountered.
            **kwargs (Any): Additional arguments.
        """
        pass

    def on_node_execute_start(self, serialized: dict[str, Any], input_data: dict[str, Any], **kwargs: Any):
        """Called when the node execute starts.

        Args:
            serialized (dict[str, Any]): Serialized node data.
            input_data (dict[str, Any]): Input data for the node.
            **kwargs (Any): Additional arguments.
        """
        pass

    def on_node_execute_end(self, serialized: dict[str, Any], output_data: dict[str, Any], **kwargs: Any):
        """Called when the node execute ends.

        Args:
            serialized (dict[str, Any]): Serialized node data.
            output_data (dict[str, Any]): Output data from the node.
            **kwargs (Any): Additional arguments.
        """
        pass

    def on_node_execute_error(self, serialized: dict[str, Any], error: BaseException, **kwargs: Any):
        """Called when the node execute errors.

        Args:
            serialized (dict[str, Any]): Serialized node data.
            error (BaseException): Error encountered.
            **kwargs (Any): Additional arguments.
        """
        pass

    def on_node_execute_run(self, serialized: dict[str, Any], **kwargs: Any):
        """Called when the node execute runs.

        Args:
            serialized (dict[str, Any]): Serialized node data.
            **kwargs (Any): Additional arguments.
        """
        pass

    def on_node_execute_stream(self, serialized: dict[str, Any], chunk: dict[str, Any] | None = None, **kwargs: Any):
        """Called when the node execute streams.

        Args:
            serialized (dict[str, Any]): Serialized node data.
            chunk (dict[str, Any] | None): Stream chunk data.
            **kwargs (Any): Additional arguments.
        """
        pass

    def on_node_skip(
        self, serialized: dict[str, Any], skip_data: dict[str, Any], input_data: dict[str, Any], **kwargs: Any
    ):
        """Called when the node skips.

        Args:
            serialized (dict[str, Any]): Serialized node data.
            skip_data (dict[str, Any]): Data related to the skip.
            input_data (dict[str, Any]): Input data for the node.
            **kwargs (Any): Additional arguments.
        """
        pass

    def on_node_canceled(self, serialized: dict[str, Any], **kwargs: Any):
        """Called when the node is canceled mid-execution.

        Args:
            serialized (dict[str, Any]): Serialized node data.
            **kwargs (Any): Additional arguments.
        """
        pass


class BaseCallbackHandler(NodeCallbackHandler, ABC):
    """Abstract base class for general callback handlers."""

    def on_workflow_start(
        self, serialized: dict[str, Any], input_data: dict[str, Any], **kwargs: Any
    ):
        """Called when the workflow starts.

        Args:
            serialized (dict[str, Any]): Serialized workflow data.
            input_data (dict[str, Any]): Input data for the workflow.
            **kwargs (Any): Additional arguments.
        """
        pass

    def on_workflow_end(
        self, serialized: dict[str, Any], output_data: dict[str, Any], **kwargs: Any
    ):
        """Called when the workflow ends.

        Args:
            serialized (dict[str, Any]): Serialized workflow data.
            output_data (dict[str, Any]): Output data from the workflow.
            **kwargs (Any): Additional arguments.
        """
        pass

    def on_workflow_error(
        self, serialized: dict[str, Any], error: BaseException, **kwargs: Any
    ):
        """Called when the workflow errors.

        Args:
            serialized (dict[str, Any]): Serialized workflow data.
            error (BaseException): Error encountered.
            **kwargs (Any): Additional arguments.
        """
        pass

    def on_workflow_canceled(self, serialized: dict[str, Any], **kwargs: Any):
        """Called when the workflow is canceled mid-execution.

        Args:
            serialized (dict[str, Any]): Serialized workflow data.
            **kwargs (Any): Additional arguments.
        """
        pass

    def on_flow_start(
        self, serialized: dict[str, Any], input_data: dict[str, Any], **kwargs: Any
    ):
        """Called when the flow starts.

        Args:
            serialized (dict[str, Any]): Serialized flow data.
            input_data (dict[str, Any]): Input data for the flow.
            **kwargs (Any): Additional arguments.
        """
        pass

    def on_flow_end(
        self, serialized: dict[str, Any], output_data: dict[str, Any], **kwargs: Any
    ):
        """Called when the flow ends.

        Args:
            serialized (dict[str, Any]): Serialized flow data.
            output_data (dict[str, Any]): Output data from the flow.
            **kwargs (Any): Additional arguments.
        """
        pass

    def on_flow_error(
        self, serialized: dict[str, Any], error: BaseException, **kwargs: Any
    ):
        """Called when the flow errors.

        Args:
            serialized (dict[str, Any]): Serialized flow data.
            error (BaseException): Error encountered.
            **kwargs (Any): Additional arguments.
        """
        pass

    def on_flow_canceled(self, serialized: dict[str, Any], **kwargs: Any):
        """Called when the flow is canceled mid-execution.

        Args:
            serialized (dict[str, Any]): Serialized flow data.
            **kwargs (Any): Additional arguments.
        """
        pass


def get_entity_id(entity_name: str, kwargs: dict) -> UUID:
    """Retrieve entity ID from kwargs.

    Args:
        entity_name (str): Name of the entity.
        kwargs (dict): Keyword arguments.

    Returns:
        UUID: Entity ID.

    Raises:
        ValueError: If entity ID is not found or invalid.
    """
    entity_id = kwargs.get(entity_name)
    if entity_name == "parent_run_id" and not entity_id:
        return entity_id
    if not entity_id:
        raise ValueError(f"{entity_name} not found")

    if isinstance(entity_id, UUID):
        return entity_id
    elif isinstance(entity_id, str):
        return UUID(entity_id)

    raise ValueError(f"{entity_name} is not UUID or str")


def get_run_id(kwargs: dict) -> UUID:
    """Retrieve run ID from kwargs.

    Args:
        kwargs (dict): Keyword arguments.

    Returns:
        UUID: Run ID.
    """
    return get_entity_id("run_id", kwargs)


def get_parent_run_id(kwargs: dict) -> UUID:
    """Retrieve parent run ID from kwargs.

    Args:
        kwargs (dict): Keyword arguments.

    Returns:
        UUID: Parent run ID.
    """
    return get_entity_id("parent_run_id", kwargs)


def get_execution_run_id(kwargs: dict) -> UUID:
    """Retrieve execution run ID from kwargs.

    Args:
        kwargs (dict): Keyword arguments.

    Returns:
        UUID: Execution run ID.
    """
    return get_entity_id("execution_run_id", kwargs)
