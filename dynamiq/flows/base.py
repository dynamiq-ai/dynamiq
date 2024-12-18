from abc import ABC
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

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
                callback.on_flow_start(self.model_dump(), input_data, **kwargs)

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
                callback.on_flow_end(self.model_dump(), output_data, **kwargs)

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
                callback.on_flow_error(self.model_dump(), error, **kwargs)
