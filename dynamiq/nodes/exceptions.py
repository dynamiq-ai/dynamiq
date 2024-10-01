from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dynamiq.nodes.node import NodeDependency


class NodeException(Exception):
    """
    Base exception class for node-related errors.

    Args:
        failed_depend (NodeDependency): The dependency that caused the exception.
        message (str, optional): Additional error message. Defaults to None.

    Attributes:
        failed_depend (NodeDependency): The dependency that caused the exception.
    """

    def __init__(self, failed_depend: "NodeDependency", message=None):
        super().__init__(message)
        self.failed_depend = failed_depend


class NodeFailedException(NodeException):
    """
    Exception raised when a node fails to execute.

    This exception is a subclass of NodeException and inherits its attributes and methods.
    """

    pass


class NodeSkippedException(NodeException):
    """
    Exception raised when a node is skipped during execution.

    This exception is a subclass of NodeException and inherits its attributes and methods.
    """

    pass


class NodeConditionFailedException(NodeException):
    """
    Exception raised when a node's condition fails to be met.

    This exception is a subclass of NodeException and inherits its attributes and methods.
    """

    pass
