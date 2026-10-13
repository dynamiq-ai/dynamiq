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


class HookBlockedException(Exception):
    """A hook ended the agent run. Not recoverable: the ReAct loop never turns it into an observation."""

    outcome: str

    def __init__(self, message: str = "", hook: str | None = None, point: str | None = None):
        super().__init__(message)
        self.message = message
        self.hook = hook
        self.point = point


class HookStopException(HookBlockedException):
    """A hook ended the run with a failure. Never retried (``retryable = False``)."""

    retryable = False
    outcome = "fail"


class HookAnswerException(HookBlockedException):
    """A hook ended the run successfully with ``message`` as the answer. Caught once, in ``Agent.execute``."""

    outcome = "answer"


class ToolBlockedException(ToolExecutionException):
    """A hook blocked a tool call: ``str(exc)`` goes back to the model as the observation."""

    outcome = "observation"

    def __init__(self, message: str, hook: str | None = None, point: str | None = None):
        super().__init__(message)
        self.hook = hook
        self.point = point


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


class OutputFileNotFoundError(RecoverableAgentException):
    """Exception raised when files listed in <output_files> do not exist on the backend."""

    pass
