import importlib

from dynamiq.nodes import Node


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

        Example:
            >>> node_class = NodeManager.get_node_by_type("LLM_OPENAI")
            >>> isinstance(node_class, type(Node))
            True
        """
        try:
            entity_module, entity_name = node_type.rsplit(".", 1)
            imported_module = importlib.import_module(entity_module)
            if entity := getattr(imported_module, entity_name, None):
                return entity
        except (ModuleNotFoundError, ImportError):
            raise ValueError(f"Node type {node_type} not found")
