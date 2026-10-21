from abc import ABC
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.callbacks.base import is_noop_hook, serialization_for
from dynamiq.runnables import Runnable, RunnableConfig
from dynamiq.utils import generate_uuid


class BaseFlow(BaseModel, Runnable, ABC):
    """
    Base class for flow implementations.

    Attributes:
        id (str): Unique identifier for the flow, generated using UUID.

    """

    id: str = Field(default_factory=generate_uuid)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **kwargs):
        """
        Initialize the BaseFlow instance.

        Args:
            **kwargs: Additional keyword arguments to be passed to the parent constructor.
        """
        super().__init__(**kwargs)
        self._results = {}

    def reset_run_state(self):
        """Reset the internal run state by clearing the results dictionary."""
        self._results = {}

    @property
    def to_dict_exclude_params(self) -> dict:
        return {}

    def to_dict(self, include_secure_params: bool = False, **kwargs) -> dict:
        """Converts the instance to a dictionary.

        Returns:
            dict: A dictionary representation of the instance.
        """
        exclude = kwargs.pop("exclude", self.to_dict_exclude_params)
        kwargs.pop("for_tracing", False)
        data = self.model_dump(
            exclude=exclude,
            serialize_as_any=kwargs.pop("serialize_as_any", True),
            **kwargs,
        )

        return data

    def run_on_flow_start(
        self, input_data: Any, config: RunnableConfig = None, **kwargs: Any
    ):
        """
        Execute callbacks when the flow starts.

        Args:
            input_data (Any): The input data for the flow.
            config (RunnableConfig, optional): Configuration for the runnable.
            **kwargs: Additional keyword arguments to be passed to the callbacks.
        """
        if config and config.callbacks:
            for callback in config.callbacks:
                if is_noop_hook(callback, "on_flow_start"):
                    continue
                for_tracing = isinstance(callback, TracingCallbackHandler)
                callback.on_flow_start(
                    serialization_for(self, callback.on_flow_start, for_tracing), input_data, **kwargs
                )

    def run_on_flow_end(
        self, output_data: Any, config: RunnableConfig = None, **kwargs: Any
    ):
        """
        Execute callbacks when the flow ends.

        Args:
            output_data (Any): The output data from the flow.
            config (RunnableConfig, optional): Configuration for the runnable.
            **kwargs: Additional keyword arguments to be passed to the callbacks.
        """
        if config and config.callbacks:
            for callback in config.callbacks:
                if is_noop_hook(callback, "on_flow_end"):
                    continue
                for_tracing = isinstance(callback, TracingCallbackHandler)
                callback.on_flow_end(serialization_for(self, callback.on_flow_end, for_tracing), output_data, **kwargs)

    def run_on_flow_error(
        self, error: BaseException, config: RunnableConfig = None, **kwargs: Any
    ):
        """
        Execute callbacks when an error occurs in the flow.

        Args:
            error (BaseException): The error that occurred during the flow execution.
            config (RunnableConfig, optional): Configuration for the runnable.
            **kwargs: Additional keyword arguments to be passed to the callbacks.
        """
        if config and config.callbacks:
            for callback in config.callbacks:
                if is_noop_hook(callback, "on_flow_error"):
                    continue
                for_tracing = isinstance(callback, TracingCallbackHandler)
                callback.on_flow_error(serialization_for(self, callback.on_flow_error, for_tracing), error, **kwargs)

    def run_on_flow_canceled(self, config: RunnableConfig = None, **kwargs: Any):
        """
        Execute callbacks when the flow is canceled mid-execution.

        Args:
            config (RunnableConfig, optional): Configuration for the runnable.
            **kwargs: Additional keyword arguments to be passed to the callbacks.
        """
        if config and config.callbacks:
            for callback in config.callbacks:
                if is_noop_hook(callback, "on_flow_canceled"):
                    continue
                for_tracing = isinstance(callback, TracingCallbackHandler)
                callback.on_flow_canceled(serialization_for(self, callback.on_flow_canceled, for_tracing), **kwargs)
