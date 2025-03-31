import inspect
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import datetime
from functools import cached_property
from queue import Empty
from typing import Any, Callable, ClassVar, Union
from uuid import uuid4

from jinja2 import Template
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, computed_field, model_validator

from dynamiq.cache.utils import cache_wf_entity
from dynamiq.callbacks import BaseCallbackHandler, NodeCallbackHandler
from dynamiq.connections import BaseConnection
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.exceptions import (
    NodeConditionFailedException,
    NodeConditionSkippedException,
    NodeException,
    NodeFailedException,
    NodeSkippedException,
)
from dynamiq.nodes.types import NodeGroup
from dynamiq.runnables import Runnable, RunnableConfig, RunnableResult, RunnableStatus
from dynamiq.storages.vector.base import BaseVectorStoreParams
from dynamiq.types.feedback import (
    ApprovalConfig,
    ApprovalInputData,
    ApprovalStreamingInputEventMessage,
    ApprovalStreamingOutputEventMessage,
    FeedbackMethod,
)
from dynamiq.types.streaming import STREAMING_EVENT, StreamingConfig, StreamingEventMessage
from dynamiq.utils import format_value, generate_uuid, merge
from dynamiq.utils.duration import format_duration
from dynamiq.utils.jsonpath import filter as jsonpath_filter
from dynamiq.utils.jsonpath import mapper as jsonpath_mapper
from dynamiq.utils.logger import logger


def ensure_config(config: RunnableConfig = None) -> RunnableConfig:
    """
    Ensure that a valid RunnableConfig is provided.

    Args:
        config (RunnableConfig, optional): The input configuration. Defaults to None.

    Returns:
        RunnableConfig: A valid RunnableConfig object.
    """
    if config is None:
        return RunnableConfig(callbacks=[])

    return config


class ErrorHandling(BaseModel):
    """
    Configuration for error handling in nodes.

    Attributes:
        timeout_seconds (float | None): Timeout in seconds for node execution.
        retry_interval_seconds (float): Interval between retries in seconds.
        max_retries (int): Maximum number of retries.
        backoff_rate (float): Rate of increase for retry intervals.
    """
    timeout_seconds: float | None = None
    retry_interval_seconds: float = 1
    max_retries: int = 0
    backoff_rate: float = 1


class Transformer(BaseModel):
    """
    Base class for input and output transformers.

    Attributes:
        path (str | None): JSONPath for data selection.
        selector (dict[str, str] | None): Mapping for data transformation.
    """
    path: str | None = None
    selector: dict[str, str] | None = None


class InputTransformer(Transformer):
    """Input transformer for nodes."""
    pass


class OutputTransformer(InputTransformer):
    """Output transformer for nodes."""
    pass


class CachingConfig(BaseModel):
    """
    Configuration for node caching.

    Attributes:
        enabled (bool): Whether caching is enabled for the node.
    """
    enabled: bool = False


class NodeReadyToRun(BaseModel):
    """
    Represents a node ready to run with its input data and dependencies.

    Attributes:
        node (Node): The node to be run.
        is_ready (bool): Whether the node is ready to run.
        input_data (Any): Input data for the node.
        depends_result (dict[str, Any]): Results of dependent nodes.
    """
    node: "Node"
    is_ready: bool
    input_data: Any = None
    depends_result: dict[str, Any] = {}

    model_config = ConfigDict(arbitrary_types_allowed=True)


class NodeDependency(BaseModel):
    """
    Represents a dependency between nodes.

    Attributes:
        node (Node): The dependent node.
        option (str | None): Optional condition for the dependency.
    """
    node: "Node"
    option: str | None = None

    def __init__(self, node: "Node", option: str | None = None):
        super().__init__(node=node, option=option)

    def to_dict(self, **kwargs) -> dict:
        """Converts the instance to a dictionary.

        Returns:
            dict: A dictionary representation of the instance.
        """
        return {"node": self.node.to_dict(**kwargs), "option": self.option}


class NodeMetadata(BaseModel):
    """
    Metadata for a node.

    Attributes:
        label (str | None): Optional label for the node.
    """
    label: str | None = None


class NodeOutputReference(BaseModel):
    """
    Represents a reference to a node output.

    Attributes:
        node (Node): The node to reference.
        output_key (str): Key for the output.
    """

    node: "Node"
    output_key: str


class NodeOutputReferences:
    """
    Provides output references for a node.

    Attributes:
        node (Node): The node to provide output references for.
    """

    def __init__(self, node: "Node"):
        self.node = node

    def __getattr__(self, key: Any):
        return NodeOutputReference(node=self.node, output_key=key)


class Node(BaseModel, Runnable, ABC):
    """
    Abstract base class for all nodes in the workflow.

    Attributes:
        id (str): Unique identifier for the node.
        name (str | None): Optional name for the node.
        group (NodeGroup): Group the node belongs to.
        description (str | None): Optional description for the node.
        error_handling (ErrorHandling): Error handling configuration.
        input_transformer (InputTransformer): Input data transformer.
        output_transformer (OutputTransformer): Output data transformer.
        caching (CachingConfig): Caching configuration.
        depends (list[NodeDependency]): List of node dependencies.
        metadata (NodeMetadata | None): Optional metadata for the node.
        is_postponed_component_init (bool): Whether component initialization is postponed.
        is_optimized_for_agents (bool): Whether to optimize output for agents. By default is set to False.
        supports_files (bool): Whether the node has access to files. By default is set to False.
    """
    id: str = Field(default_factory=generate_uuid)
    name: str | None = None
    description: str | None = None
    group: NodeGroup
    error_handling: ErrorHandling = Field(default_factory=ErrorHandling)
    input_transformer: InputTransformer = Field(default_factory=InputTransformer)
    input_mapping: dict[str, Any] = {}
    output_transformer: OutputTransformer = Field(default_factory=OutputTransformer)
    caching: CachingConfig = Field(default_factory=CachingConfig)
    streaming: StreamingConfig = Field(default_factory=StreamingConfig)
    approval: ApprovalConfig = Field(default_factory=ApprovalConfig)

    depends: list[NodeDependency] = []
    metadata: NodeMetadata | None = None

    is_postponed_component_init: bool = False
    is_optimized_for_agents: bool = False
    is_files_allowed: bool = False

    _output_references: NodeOutputReferences = PrivateAttr()

    model_config = ConfigDict(arbitrary_types_allowed=True)
    input_schema: ClassVar[type[BaseModel] | None] = None
    callbacks: list[NodeCallbackHandler] = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.is_postponed_component_init:
            self.init_components()

        self._output_references = NodeOutputReferences(node=self)

    @computed_field
    @cached_property
    def type(self) -> str:
        return f"{self.__module__.rsplit('.', 1)[0]}.{self.__class__.__name__}"

    @staticmethod
    def _validate_dependency_status(depend: NodeDependency, depends_result: dict[str, RunnableResult]):
        """
        Validate the status of a dependency.

        Args:
            depend (NodeDependency): The dependency to validate.
            depends_result (dict[str, RunnableResult]): Results of dependent nodes.

        Raises:
            NodeException: If the dependency result is missing.
            NodeFailedException: If the dependency failed.
            NodeSkippedException: If the dependency was skipped.
        """
        if not (dep_result := depends_result.get(depend.node.id)):
            raise NodeException(
                failed_depend=depend,
                message=f"Dependency {depend.node.id}: result missed",
            )

        if dep_result.status == RunnableStatus.FAILURE:
            raise NodeFailedException(
                failed_depend=depend, message=f"Dependency {depend.node.id}: failed"
            )

        if dep_result.status == RunnableStatus.SKIP:
            raise NodeSkippedException(failed_depend=depend, message=f"Dependency {depend.node.id}: skipped")

    @staticmethod
    def _validate_dependency_condition(depend: NodeDependency, depends_result: dict[str, RunnableResult]):
        """
        Validate the condition of a dependency.

        Args:
            depend (NodeDependency): The dependency to validate.
            depends_result (dict[str, RunnableResult]): Results of dependent nodes.

        Raises:
            NodeConditionFailedException: If the dependency condition is not met.
            NodeConditionSkippedException: If the dependency condition is skipped.
        """
        if (
            (dep_output_data := depends_result.get(depend.node.id))
            and (isinstance(dep_output_data.output, dict))
            and (dep_condition_result := dep_output_data.output.get(depend.option))
        ):
            if dep_condition_result.status == RunnableStatus.FAILURE:
                raise NodeConditionFailedException(
                    failed_depend=depend,
                    message=f"Dependency {depend.node.id} condition {depend.option}: result is false",
                )
            if dep_condition_result.status == RunnableStatus.SKIP:
                raise NodeConditionSkippedException(
                    failed_depend=depend,
                    message=f"Dependency {depend.node.id} condition {depend.option}: skipped",
                )

    @staticmethod
    def _validate_input_mapping_value_func(func: Callable):
        """
        Validate input mapping value function.

        Args:
            func (Callable): Input mapping value function.

        Raises:
            ValueError: If the function does not accept 'inputs' and 'outputs' or **kwargs.
        """
        params = inspect.signature(func).parameters

        # Check if the function accepts the at least 'inputs' and 'outputs' parameters
        if len(params) >= 2:
            return

        # Check if the function accepts **kwargs
        elif params and list(params.values())[0].kind == inspect.Parameter.VAR_KEYWORD:
            return

        raise ValueError(f"Input function '{func.__name__}' must accept parameters 'inputs' and 'outputs' or **kwargs.")

    def validate_depends(self, depends_result):
        """
        Validate all dependencies of the node.

        Args:
            depends_result (dict): Results of dependent nodes.

        Raises:
            Various exceptions based on dependency validation results.
        """
        for dep in self.depends:
            self._validate_dependency_status(depend=dep, depends_result=depends_result)
            if dep.option:
                self._validate_dependency_condition(
                    depend=dep, depends_result=depends_result
                )

    def validate_input_schema(self, input_data: dict[str, Any], **kwargs) -> dict[str, Any] | BaseModel:
        """
        Validate input data against the input schema. Returns instance of input_schema if it is is provided.

        Args:
            input_data (Any): Input data to validate.

        Raises:
            NodeException: If input data does not match the input schema.
        """
        from dynamiq.nodes.agents.exceptions import RecoverableAgentException

        if self.input_schema:
            try:
                return self.input_schema.model_validate(
                    input_data, context=kwargs | self.get_context_for_input_schema()
                )
            except Exception as e:
                if kwargs.get("recoverable_error", False):
                    raise RecoverableAgentException(f"Input data validation failed: {e}")
                raise e

        return input_data

    def transform_input(
        self, input_data: dict, depends_result: dict[Any, RunnableResult], use_input_transformer: bool = True, **kwargs
    ) -> dict:
        """
        Transform input data for the node.

        Args:
            input_data (dict): Input data for the node.
            depends_result (dict): Results of dependent nodes.
            use_input_transformer (bool): Determines if InputTransformer will be applied to the input.

        Raises:
            NodeException: If a dependency result is missing or input mapping fails.

        Returns:
            dict: Transformed input data.
        """
        # Apply input transformer
        if (self.input_transformer.path or self.input_transformer.selector) and use_input_transformer:
            depends_result_as_dict = {k: result.to_depend_dict() for k, result in depends_result.items()}
            inputs = self.transform(input_data | depends_result_as_dict, self.input_transformer, self.id)
        else:
            inputs = input_data | {k: result.to_tracing_depend_dict() for k, result in depends_result.items()}

        # Apply input bindings
        for key, value in self.input_mapping.items():
            if isinstance(value, NodeOutputReference):
                depend_result = depends_result.get(value.node.id)
                if not depend_result:
                    raise NodeException(message=f"Dependency {value.node.id}: result not found.")
                if value.output_key not in depend_result.output:
                    raise NodeException(message=f"Dependency {value.node.id} output {value.output_key}: not found.")

                inputs[key] = depend_result.output[value.output_key]

            elif callable(value):
                try:
                    inputs[key] = value(inputs, {d_id: result.output for d_id, result in depends_result.items()})
                except Exception:
                    raise NodeException(message=f"Input mapping {key}: failed.")
            else:
                inputs[key] = value

        return inputs

    def init_components(self, connection_manager: ConnectionManager | None = None):
        """
        Initialize node components.

        Args:
            connection_manager (ConnectionManager, optional): The connection manager.
        """
        self.is_postponed_component_init = False

    @staticmethod
    def transform(data: Any, transformer: Transformer, node_id: str) -> Any:
        """
        Apply transformation to data.

        Args:
            data (Any): Input data to transform.
            transformer (Transformer): Transformer to apply.
            node_id (str): ID of the node performing the transformation.

        Returns:
            Any: Transformed data.
        """
        output = jsonpath_filter(data, transformer.path, node_id)
        output = jsonpath_mapper(output, transformer.selector, node_id)
        return output

    def transform_output(self, output_data: Any) -> Any:
        """
        Transform output data from the node.

        Args:
            output_data (Any): Output data to transform.

        Returns:
            Any: Transformed output data.
        """
        return self.transform(output_data, self.output_transformer, self.id)

    @property
    def to_dict_exclude_params(self):
        return {
            "client": True,
            "vector_store": True,
            "depends": True,
            "input_mapping": True,
        }

    @property
    def to_dict_exclude_secure_params(self):
        return self.to_dict_exclude_params | {"connection": {"api_key": True}}

    def to_dict(self, include_secure_params: bool = False, **kwargs) -> dict:
        """Converts the instance to a dictionary.

        Returns:
            dict: A dictionary representation of the instance.
        """
        exclude = kwargs.pop(
            "exclude", self.to_dict_exclude_params if include_secure_params else self.to_dict_exclude_secure_params
        )
        data = self.model_dump(
            exclude=exclude,
            serialize_as_any=kwargs.pop("serialize_as_any", True),
            **kwargs,
        )
        data["depends"] = [depend.to_dict(**kwargs) for depend in self.depends]
        data["input_mapping"] = format_value(self.input_mapping)[0]
        return data

    def send_streaming_approval_message(
        self, template: str, input_data: dict, approval_config: ApprovalConfig, config: RunnableConfig = None, **kwargs
    ) -> ApprovalInputData:
        """
        Sends approval message and waits for response.

        Args:
            template (str): Template to send.
            input_data (dict): Data that will be sent.
            approval_config (ApprovalConfig): Configuration for approval.
            config (RunnableConfig, optional): Configuration for the runnable.
            **kwargs: Additional keyword arguments.

        Return:
            ApprovalInputData: Response to approval message.

        """
        event = ApprovalStreamingOutputEventMessage(
            wf_run_id=config.run_id,
            entity_id=self.id,
            data={"template": template, "data": input_data, "mutable_data_params": approval_config.mutable_data_params},
            event=approval_config.event,
        )

        logger.info(f"Node {self.name} - {self.id}: sending approval.")

        self.run_on_node_execute_stream(callbacks=config.callbacks, event=event, **kwargs)

        output: ApprovalInputData = self.get_input_streaming_event(
            event=approval_config.event, event_msg_type=ApprovalStreamingInputEventMessage, config=config
        ).data

        return output

    def send_console_approval_message(self, template: str) -> ApprovalInputData:
        """
        Sends approval message in console and waits for response.

        Args:
            template (dict): Template to send.
        Returns:
            ApprovalInputData: Response to approval message.
        """
        feedback = input(template)
        return ApprovalInputData(feedback=feedback)

    def send_approval_message(
        self, approval_config: ApprovalConfig, input_data: dict, config: RunnableConfig = None, **kwargs
    ) -> ApprovalInputData:
        """
        Sends approval message and determines if it was approved or disapproved (canceled).

        Args:
            approval_config (ApprovalConfig): Configuration for the approval.
            input_data (dict): Data that will be sent.
            config (RunnableConfig, optional): Configuration for the runnable.
            **kwargs: Additional keyword arguments.

        Returns:
            ApprovalInputData: Result of approval.
        """

        message = Template(approval_config.msg_template).render(self.to_dict(), input_data=input_data)
        match approval_config.feedback_method:
            case FeedbackMethod.STREAM:
                approval_result = self.send_streaming_approval_message(
                    message, input_data, approval_config, config=config, **kwargs
                )
            case FeedbackMethod.CONSOLE:
                approval_result = self.send_console_approval_message(message)
            case _:
                raise ValueError(f"Error: Incorrect feedback method is chosen {approval_config.feedback_method}.")

        update_params = {
            feature_name: approval_result.data[feature_name]
            for feature_name in approval_config.mutable_data_params
            if feature_name in approval_result.data
        }
        approval_result.data = {**input_data, **update_params}

        if approval_result.is_approved is None:
            if approval_result.feedback == approval_config.accept_pattern:
                logger.info(
                    f"Node {self.name} action was approved by human "
                    f"with provided feedback '{approval_result.feedback}'."
                )
                approval_result.is_approved = True

            else:
                approval_result.is_approved = False
                logger.info(
                    f"Node {self.name} action was canceled by human"
                    f"with provided feedback '{approval_result.feedback}'."
                )

        return approval_result

    def get_approved_data_or_origin(
        self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs
    ) -> dict[str, Any]:
        """
        Approves or disapproves (cancels) Node execution by requesting feedback.
        Updates input data according to the feedback or leaves it the same.
        Raises NodeException if execution was canceled by feedback.

        Args:
            input_data(dict[str, Any]): Input data.
            config (RunnableConfig, optional): Configuration for the runnable.
            **kwargs: Additional keyword arguments.

        Returns:
            dict[str, Any]: Updated input data.

        Raises:
            NodeException: If Node execution was canceled by feedback.
        """
        if self.approval.enabled:
            approval_result = self.send_approval_message(self.approval, input_data, config=config, **kwargs)
            if not approval_result.is_approved:
                raise NodeException(
                    message=f"Execution was canceled by human with feedback {approval_result.feedback}",
                    recoverable=True,
                    failed_depend=NodeDependency(self, option="Execution was canceled."),
                )
            return approval_result.data

        return input_data

    def run(
        self,
        input_data: Any,
        config: RunnableConfig = None,
        depends_result: dict = None,
        **kwargs,
    ) -> RunnableResult:
        """
        Run the node with given input data and configuration.

        Args:
            input_data (Any): Input data for the node.
            config (RunnableConfig, optional): Configuration for the run. Defaults to None.
            depends_result (dict, optional): Results of dependent nodes. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            RunnableResult: Result of the node execution.
        """
        from dynamiq.nodes.agents.exceptions import RecoverableAgentException

        logger.info(f"Node {self.name} - {self.id}: execution started.")
        transformed_input = input_data
        time_start = datetime.now()

        config = ensure_config(config)

        run_id = uuid4()
        merged_kwargs = merge(kwargs, {"run_id": run_id, "parent_run_id": kwargs.get("parent_run_id", run_id)})
        if depends_result is None:
            depends_result = {}

        try:
            try:
                self.validate_depends(depends_result)
                input_data = self.get_approved_data_or_origin(input_data, config=config, **merged_kwargs)
            except NodeException as e:
                transformed_input = input_data | {
                    k: result.to_tracing_depend_dict() for k, result in depends_result.items()
                }
                skip_data = {"failed_dependency": e.failed_depend.to_dict()}
                self.run_on_node_skip(
                    callbacks=config.callbacks,
                    skip_data=skip_data,
                    input_data=transformed_input,
                    **merged_kwargs,
                )
                logger.info(f"Node {self.name} - {self.id}: execution skipped.")
                return RunnableResult(
                    status=RunnableStatus.SKIP,
                    input=transformed_input,
                    output=format_value(e, recoverable=e.recoverable)[0],
                )

            transformed_input = self.transform_input(input_data=input_data, depends_result=depends_result, **kwargs)
            self.run_on_node_start(config.callbacks, transformed_input, **merged_kwargs)
            cache = cache_wf_entity(
                entity_id=self.id,
                cache_enabled=self.caching.enabled,
                cache_config=config.cache,
            )

            output, from_cache = cache(self.execute_with_retry)(
                self.validate_input_schema(transformed_input, **kwargs), config, **merged_kwargs
            )

            merged_kwargs["is_output_from_cache"] = from_cache
            transformed_output = self.transform_output(output)

            self.run_on_node_end(config.callbacks, transformed_output, **merged_kwargs)

            logger.info(
                f"Node {self.name} - {self.id}: execution succeeded in "
                f"{format_duration(time_start, datetime.now())}."
            )
            return RunnableResult(status=RunnableStatus.SUCCESS, input=transformed_input, output=transformed_output)
        except Exception as e:
            self.run_on_node_error(callbacks=config.callbacks, error=e, input_data=transformed_input, **merged_kwargs)
            logger.error(
                f"Node {self.name} - {self.id}: execution failed in {e}"
                f"{format_duration(time_start, datetime.now())}."
            )

            recoverable = isinstance(e, RecoverableAgentException)
            return RunnableResult(
                status=RunnableStatus.FAILURE,
                input=input_data,
                output=format_value(e, recoverable=recoverable)[0],
            )

    def execute_with_retry(self, input_data: dict[str, Any] | BaseModel, config: RunnableConfig = None, **kwargs):
        """
        Execute the node with retry logic.

        Args:
            input_data (dict[str, Any]): Input data for the node.
            config (RunnableConfig, optional): Configuration for the execution. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: Result of the node execution.

        Raises:
            Exception: If all retry attempts fail.
        """
        config = ensure_config(config)

        error = None
        n_attempt = self.error_handling.max_retries + 1
        for attempt in range(n_attempt):
            merged_kwargs = merge(kwargs, {"execution_run_id": uuid4()})

            self.run_on_node_execute_start(config.callbacks, input_data, **merged_kwargs)

            try:
                output = self.execute_with_timeout(
                    self.error_handling.timeout_seconds,
                    input_data,
                    config,
                    **merged_kwargs,
                )

                self.run_on_node_execute_end(config.callbacks, output, **merged_kwargs)
                return output
            except TimeoutError as e:
                error = e
                self.run_on_node_execute_error(config.callbacks, error, **merged_kwargs)
                logger.warning(f"Node {self.name} - {self.id}: timeout.")
            except Exception as e:
                error = e
                self.run_on_node_execute_error(config.callbacks, error, **merged_kwargs)
                logger.error(f"Node {self.name} - {self.id}: execution error: {e}")

            # do not sleep after the last attempt
            if attempt < n_attempt - 1:
                time_to_sleep = self.error_handling.retry_interval_seconds * (
                    self.error_handling.backoff_rate**attempt
                )
                logger.info(
                    f"Node {self.name} - {self.id}: retrying in {time_to_sleep} seconds."
                )
                time.sleep(time_to_sleep)

        logger.error(
            f"Node {self.name} - {self.id}: execution failed after {n_attempt} attempts."
        )
        raise error

    def execute_with_timeout(
        self,
        timeout: float | None,
        input_data: dict[str, Any] | BaseModel,
        config: RunnableConfig = None,
        **kwargs,
    ):
        """
        Execute the node with a timeout.

        Args:
            timeout (float | None): Timeout duration in seconds.
            input_data (dict[str, Any]): Input data for the node.
            config (RunnableConfig, optional): Configuration for the runnable.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: Result of the execution.

        Raises:
            Exception: If execution fails or times out.
        """
        with ThreadPoolExecutor() as executor:
            future = executor.submit(self.execute, input_data, config=config, **kwargs)

            try:
                result = future.result(timeout=timeout)
            except Exception as e:
                raise e

            return result

    def get_context_for_input_schema(self) -> dict:
        """Provides context for input schema that is required for proper validation."""
        return {}

    def get_input_streaming_event(
        self,
        event_msg_type: "type[StreamingEventMessage]" = StreamingEventMessage,
        event: str | None = None,
        config: RunnableConfig = None,
    ) -> StreamingEventMessage:
        """
        Get the input streaming event from the input streaming.

        Args:
            event_msg_type (Type[StreamingEventMessage], optional): The event message type to use.
            event (str, optional): The event to use for the message.
            config (RunnableConfig, optional): Configuration for the runnable.
        """
        # Use runnable streaming configuration. If not found use node streaming configuration
        streaming = getattr(config.nodes_override.get(self.id), "streaming", None) or self.streaming
        if streaming.input_streaming_enabled:
            while not streaming.input_queue_done_event or not streaming.input_queue_done_event.is_set():
                try:
                    data = streaming.input_queue.get(timeout=streaming.timeout)
                except Empty:
                    raise ValueError(f"Input streaming timeout: {streaming.timeout} exceeded.")

                try:
                    event_msg = event_msg_type.model_validate_json(data)
                    if event and event_msg.event != event:
                        raise ValueError()
                except ValueError:
                    logger.error(
                        f"Invalid streaming event data: {data}. "
                        f"Allowed event: {event}, event_msg_type: {event_msg_type}"
                    )
                    continue

                return event_msg

        raise ValueError("Input streaming is not enabled.")

    def run_on_node_start(
        self,
        callbacks: list[BaseCallbackHandler],
        input_data: dict[str, Any],
        **kwargs,
    ) -> None:
        """
        Run callbacks on node start.

        Args:
            callbacks (list[BaseCallbackHandler]): List of callback handlers.
            input_data (dict[str, Any]): Input data for the node.
            **kwargs: Additional keyword arguments.
        """

        for callback in callbacks + self.callbacks:
            try:
                callback.on_node_start(self.to_dict(), input_data, **kwargs)
            except Exception as e:
                logger.error(f"Error running callback {callback.__class__.__name__}: {e}")

    def run_on_node_end(
        self,
        callbacks: list[BaseCallbackHandler],
        output_data: dict[str, Any],
        **kwargs,
    ) -> None:
        """
        Run callbacks on node end.

        Args:
            callbacks (list[BaseCallbackHandler]): List of callback handlers.
            output_data (dict[str, Any]): Output data from the node.
            **kwargs: Additional keyword arguments.
        """
        for callback in callbacks + self.callbacks:
            try:
                callback.on_node_end(self.model_dump(), output_data, **kwargs)
            except Exception as e:
                logger.error(f"Error running callback {callback.__class__.__name__}: {e}")

    def run_on_node_error(
        self,
        callbacks: list[BaseCallbackHandler],
        error: BaseException,
        **kwargs,
    ) -> None:
        """
        Run callbacks on node error.

        Args:
            callbacks (list[BaseCallbackHandler]): List of callback handlers.
            error (BaseException): The error that occurred.
            **kwargs: Additional keyword arguments.
        """
        for callback in callbacks + self.callbacks:
            try:
                callback.on_node_error(self.to_dict(), error, **kwargs)
            except Exception as e:
                logger.error(f"Error running callback {callback.__class__.__name__}: {e}")

    def run_on_node_skip(
        self,
        callbacks: list[BaseCallbackHandler],
        skip_data: dict[str, Any],
        input_data: dict[str, Any],
        **kwargs,
    ) -> None:
        """
        Run callbacks on node skip.

        Args:
            callbacks (list[BaseCallbackHandler]): List of callback handlers.
            skip_data (dict[str, Any]): Data related to the skip.
            input_data (dict[str, Any]): Input data for the node.
            **kwargs: Additional keyword arguments.
        """
        for callback in callbacks + self.callbacks:
            try:
                callback.on_node_skip(self.to_dict(), skip_data, input_data, **kwargs)
            except Exception as e:
                logger.error(f"Error running callback {callback.__class__.__name__}: {e}")

    def run_on_node_execute_start(
        self,
        callbacks: list[BaseCallbackHandler],
        input_data: dict[str, Any] | BaseModel,
        **kwargs,
    ) -> None:
        """
        Run callbacks on node execute start.

        Args:
            callbacks (list[BaseCallbackHandler]): List of callback handlers.
            input_data (dict[str, Any]): Input data for the node.
            **kwargs: Additional keyword arguments.
        """
        if isinstance(input_data, BaseModel):
            input_data = dict(input_data)

        for callback in callbacks + self.callbacks:
            try:
                callback.on_node_execute_start(self.to_dict(), input_data, **kwargs)
            except Exception as e:
                logger.error(f"Error running callback {callback.__class__.__name__}: {e}")

    def run_on_node_execute_end(
        self,
        callbacks: list[BaseCallbackHandler],
        output_data: dict[str, Any],
        **kwargs,
    ) -> None:
        """
        Run callbacks on node execute end.

        Args:
            callbacks (list[BaseCallbackHandler]): List of callback handlers.
            output_data (dict[str, Any]): Output data from the node.
            **kwargs: Additional keyword arguments.
        """
        for callback in callbacks + self.callbacks:
            try:
                callback.on_node_execute_end(self.to_dict(), output_data, **kwargs)
            except Exception as e:
                logger.error(f"Error running callback {callback.__class__.__name__}: {e}")

    def run_on_node_execute_error(
        self,
        callbacks: list[BaseCallbackHandler],
        error: BaseException,
        **kwargs,
    ) -> None:
        """
        Run callbacks on node execute error.

        Args:
            callbacks (list[BaseCallbackHandler]): List of callback handlers.
            error (BaseException): The error that occurred.
            **kwargs: Additional keyword arguments.
        """
        for callback in callbacks + self.callbacks:
            try:
                callback.on_node_execute_error(self.model_dump(), error, **kwargs)
            except Exception as e:
                logger.error(f"Error running callback {callback.__class__.__name__}: {e}")

    def run_on_node_execute_run(
        self,
        callbacks: list[BaseCallbackHandler],
        **kwargs,
    ) -> None:
        """
        Run callbacks on node execute run.

        Args:
            callbacks (list[BaseCallbackHandler]): List of callback handlers.
            **kwargs: Additional keyword arguments.
        """
        for callback in callbacks + self.callbacks:
            try:
                callback.on_node_execute_run(self.to_dict(), **kwargs)
            except Exception as e:
                logger.error(f"Error running callback {callback.__class__.__name__}: {e}")

    def run_on_node_execute_stream(
        self,
        callbacks: list[BaseCallbackHandler],
        chunk: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """
        Run callbacks on node execute stream.

        Args:
            callbacks (list[BaseCallbackHandler]): List of callback handlers.
            chunk (dict[str, Any]): Chunk of streaming data.
            **kwargs: Additional keyword arguments.
        """
        for callback in callbacks + self.callbacks:
            try:
                callback.on_node_execute_stream(self.to_dict(), chunk, **kwargs)
            except Exception as e:
                logger.error(f"Error running callback {callback.__class__.__name__}: {e}")

    @abstractmethod
    def execute(self, input_data: dict[str, Any] | BaseModel, config: RunnableConfig = None, **kwargs) -> Any:
        """
        Execute the node with the given input.
        Args:
            input_data (dict[str, Any]): Input data for the node.
            config (RunnableConfig, optional): Configuration for the runnable.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: Result of the execution.
        """
        pass

    def depends_on(self, nodes: Union["Node", list["Node"]]):
        """
        Add dependencies for this node. Accepts either a single node or a list of nodes.

        Args:
            nodes (Node or list[Node]): A single node or list of nodes this node depends on.

        Raises:
            TypeError: If the input is neither a Node nor a list of Node instances.
            ValueError: If an empty list is provided.

        Returns:
            self: Enables method chaining.
        """

        if nodes is None:
            raise ValueError("Nodes cannot be None.")

        # If a single node is provided, convert it to a list
        if isinstance(nodes, Node):
            nodes = [nodes]

        # Ensure the input is a list of Node instances
        if not isinstance(nodes, list) or not all(isinstance(node, Node) for node in nodes):
            raise TypeError(f"Expected a Node or a list of Node instances, but got {type(nodes).__name__}.")

        if not nodes:
            raise ValueError("Cannot add an empty list of dependencies.")

        # Add each node as a dependency
        for node in nodes:
            self.depends.append(NodeDependency(node))

        return self  # enable chaining

    def enable_streaming(self, event: str = STREAMING_EVENT):
        """
        Enable streaming for the node and optionally set the event name.

        Args:
            event (str): The event name for streaming. Defaults to 'streaming'.

        Returns:
            self: Enables method chaining.
        """
        self.streaming.enabled = True
        self.streaming.event = event
        return self

    @property
    def outputs(self):
        """
        Provide the output references for the node.
        """
        return self._output_references

    def inputs(self, **kwargs):
        """
        Add input mappings for the node.

        Returns:
            self: Enables method chaining.

        Examples:
            from dynamiq.nodes.llms import OpenAI

            openai_1_node = OpenAI(...)
            openai_2_node = OpenAI(...)
            openai_3_node = OpenAI(...)

            def merge_and_short_content(inputs: dict, outputs: dict[str, dict]):
                return (
                    f"- {outputs[openai_1_node.id]['content'][:200]} \n - {outputs[openai_2_node.id]['content'][:200]}"
                )

            openai_4_node = (
                OpenAI(
                    ...
                    prompt=prompts.Prompt(
                        messages=[
                            prompts.Message(
                                role="user",
                                content=(
                                    "Please simplify that information for {{purpose}}:\n"
                                    "{{extra_instructions}}\n"
                                    "{{content}}\n"
                                    "{{extra_content}}"
                                ),
                            )
                        ],
                    ),
                )
                .inputs(
                    purpose="10 years old kids",
                    extra_instructions="Please return information in readable format.",
                    content=merge_and_short_content,
                    extra_content=openai_3_node.outputs.content,
                )
                .depends_on([openai_1_node, openai_2_node, openai_3_node])
            )
        """
        for key, value in kwargs.items():
            if callable(value):
                self._validate_input_mapping_value_func(value)

            self.input_mapping[key] = value
        return self

    def deep_merge(self, source: dict, destination: dict) -> dict:
        """
        Recursively merge dictionaries with proper override behavior.

        Args:
            source: Source dictionary with higher priority values
            destination: Destination dictionary with lower priority values

        Returns:
            dict: Merged dictionary where source values override destination values,
                  and lists are concatenated when both source and destination have lists
        """
        result = destination.copy()
        for key, value in source.items():
            if key in result:
                if isinstance(value, dict) and isinstance(result[key], dict):
                    result[key] = self.deep_merge(value, result[key])
                elif isinstance(value, list) and isinstance(result[key], list):
                    result[key] = result[key] + value
                else:
                    result[key] = value
            else:
                result[key] = value
        return result


class ConnectionNode(Node, ABC):
    """
    Abstract base class for nodes that require a connection.

    Attributes:
        connection (BaseConnection | None): The connection to use.
        client (Any | None): The client instance.
    """

    connection: BaseConnection | None = None
    client: Any | None = None

    @model_validator(mode="after")
    def validate_connection_client(self):
        """Validate that either connection or client is specified."""
        if not self.client and not self.connection:
            raise ValueError("'connection' or 'client' should be specified")
        return self

    def init_components(self, connection_manager: ConnectionManager | None = None):
        """
        Initialize components for the node.

        Args:
            connection_manager (ConnectionManager, optional): The connection manager. Defaults to ConnectionManager.
        """
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        if self.client is None:
            self.client = connection_manager.get_connection_client(
                connection=self.connection
            )


class VectorStoreNode(ConnectionNode, BaseVectorStoreParams, ABC):
    vector_store: Any | None = None

    @model_validator(mode="after")
    def validate_connection_client(self):
        if not self.vector_store and not self.connection:
            raise ValueError("'connection' or 'vector_store' should be specified")
        return self

    @property
    @abstractmethod
    def vector_store_cls(self):
        raise NotImplementedError

    @property
    def vector_store_params(self):
        return self.model_dump(include=set(BaseVectorStoreParams.model_fields)) | {
            "connection": self.connection,
            "client": self.client,
        }

    def connect_to_vector_store(self):
        vector_store_params = self.vector_store_params
        vector_store = self.vector_store_cls(**vector_store_params)

        logger.debug(
            f"Node {self.name} - {self.id}: connected to {self.vector_store_cls.__name__} vector store with"
            f" {vector_store_params}"
        )

        return vector_store

    def init_components(self, connection_manager: ConnectionManager | None = None):
        """
        Initialize components for the node.

        Args:
            connection_manager (ConnectionManager, optional): The connection manager. Defaults to ConnectionManager.
        """
        connection_manager = connection_manager or ConnectionManager()
        # Use vector_store client if it is already initialized
        if self.vector_store:
            self.client = self.vector_store.client

        super().init_components(connection_manager)

        if self.vector_store is None:
            self.vector_store = self.connect_to_vector_store()
