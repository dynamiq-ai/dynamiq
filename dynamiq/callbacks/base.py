from abc import ABC
from typing import Any
from uuid import UUID


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
