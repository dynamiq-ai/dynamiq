class RecoverableAgentException(Exception):
    """
    Base exception class for recoverable agent errors.
    """

    def __init__(self, *args, recoverable: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.recoverable = recoverable


class ActionParsingException(RecoverableAgentException):
    """
    Exception raised when an action cannot be parsed. Raising this exeption will allow Agent to reiterate.

    This exception is a subclass of AgentException and inherits its attributes and methods.
    """

    pass


class AgentUnknownToolException(RecoverableAgentException):
    """
    Exception raised when a unknown tool is requested. Raising this exeption will allow Agent to reiterate.

    This exception is a subclass of AgentException and inherits its attributes and methods.
    """

    pass


class ToolExecutionException(RecoverableAgentException):
    """
    Exception raised when a tools fails to execute. Raising this exeption will allow Agent to reiterate.

    This exception is a subclass of AgentException and inherits its attributes and methods.
    """

    pass


class InvalidActionException(RecoverableAgentException):
    """
    Exception raised when invalid action is chosen. Raising this exeption will allow Agent to reiterate.

    This exception is a subclass of AgentException and inherits its attributes and methods.
    """

    pass


class MaxLoopsExceededException(RecoverableAgentException):
    """
    Exception raised when the agent exceeds the maximum number of allowed loops.

    This exception is recoverable, meaning the agent can continue after catching this exception.
    """

    def __init__(
        self, message: str = "Maximum number of loops reached without finding a final answer.", recoverable: bool = True
    ):
        super().__init__(message, recoverable=recoverable)
