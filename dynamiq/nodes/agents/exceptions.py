from dynamiq.auth import AuthRequest


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


class ToolAuthRequiredException(RecoverableAgentException):
    """
    Exception raised when a tool requires authentication before continuing.
    Carries the corresponding AuthRequest payload for client handling.
    """

    def __init__(self, auth_request: AuthRequest, message: str | None = None, **kwargs):
        self.auth_request = auth_request
        super().__init__(message or "Tool requires authentication before it can continue.", **kwargs)


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


class ParsingError(RecoverableAgentException):
    """Base class for parsing errors."""

    pass


class XMLParsingError(ParsingError):
    """Exception raised when XML structure is invalid or cannot be parsed."""

    pass


class TagNotFoundError(ParsingError):
    """Exception raised when required XML tags are missing."""

    pass


class JSONParsingError(ParsingError):
    """Exception raised when expected JSON content within XML is invalid."""

    pass
