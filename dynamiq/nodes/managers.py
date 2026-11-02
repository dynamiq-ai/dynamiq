from dynamiq.nodes import Node
from dynamiq.utils.lazy_imports import OptionalDependencyError, import_type


class NodeManager:
    """A class for managing and retrieving node types."""

    @staticmethod
    def get_node_by_type(node_type: str) -> type[Node]:
        """
        Retrieves a node class based on the given node type.

        Args:
            node_type (str): The type of node to retrieve.

        Returns:
            type[Node]: The node class corresponding to the given type.

        Raises:
            ValueError: If the node type is not found.
            OptionalDependencyError: If the node type needs an optional dependency that is not installed.

        Example:
            >>> node_class = NodeManager.get_node_by_type("LLM_OPENAI")
            >>> isinstance(node_class, type(Node))
            True
        """
        try:
            if entity := import_type(node_type):
                return entity
        except OptionalDependencyError:
            raise
        except (ModuleNotFoundError, ImportError, AttributeError) as e:
            raise ValueError(f"Node type {node_type} not found") from e
