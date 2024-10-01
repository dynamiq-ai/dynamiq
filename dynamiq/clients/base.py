import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dynamiq.callbacks.tracing import Run


class BaseTracingClient(abc.ABC):
    """Abstract base class for tracing clients."""

    @abc.abstractmethod
    def trace(self, runs: list["Run"]) -> None:
        """Trace the given runs.

        Args:
            runs (list["Run"]): List of runs to trace.

        Raises:
            NotImplementedError: If not implemented.
        """
        raise NotImplementedError
