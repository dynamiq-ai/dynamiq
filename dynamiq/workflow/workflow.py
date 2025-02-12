from datetime import datetime
from os import PathLike
from typing import IO, TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import BaseModel, Field

from dynamiq.connections.managers import ConnectionManager
from dynamiq.flows import BaseFlow, Flow
from dynamiq.runnables import Runnable, RunnableConfig, RunnableResult, RunnableStatus
from dynamiq.utils import format_duration, generate_uuid, merge
from dynamiq.utils.logger import logger

if TYPE_CHECKING:
    from dynamiq.serializers.loaders.yaml import WorkflowYamlData


class Workflow(BaseModel, Runnable):
    """
    A container for a flow that manages its lifecycle, YAML serialization,
    versioning, metadata, callbacks, and configuration.

    Attributes:
        id (str): Unique identifier for the workflow.
        flow (BaseFlow): The flow associated with the workflow.
        version (str | None): Version of the workflow.
    """
    id: str = Field(default_factory=generate_uuid)
    flow: BaseFlow = Field(default_factory=Flow)
    version: str | None = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    @classmethod
    def from_yaml_file(
        cls,
        file_path: str,
        wf_id: str = None,
        connection_manager: ConnectionManager | None = None,
        init_components: bool = False,
    ):
        """Load workflow from a YAML file.

        Args:
            file_path (str): Path to the YAML file.
            wf_id (str, optional): Workflow ID. Defaults to None.
            connection_manager (ConnectionManager | None, optional): Connection manager. Defaults to None.
            init_components (bool, optional): Whether to initialize components. Defaults to False.

        Returns:
            Workflow: Loaded workflow instance.
        """
        from dynamiq.serializers.loaders.yaml import WorkflowYAMLLoader

        try:
            wf_data = WorkflowYAMLLoader.load(
                file_path, connection_manager, init_components
            )
        except Exception as e:
            logger.error(f"Failed to load workflow from YAML. {e}")
            raise

        return cls.from_yaml_file_data(wf_data, wf_id)

    @classmethod
    def from_yaml_file_data(cls, file_data: "WorkflowYamlData", wf_id: str = None):
        """Load workflow from YAML file data.

        Args:
            file_data (WorkflowYamlData): Data loaded from the YAML file.
            wf_id (str, optional): Workflow ID. Defaults to None.

        Returns:
            Workflow: Loaded workflow instance.
        """
        try:
            if wf_id is None:
                if len(file_data.workflows) > 1:
                    raise ValueError(
                        "Multiple workflows found in YAML. Please specify 'wf_id'"
                    )
                return list(file_data.workflows.values())[0]

            if wf := file_data.workflows.get(wf_id):
                return wf
            else:
                raise ValueError(f"Workflow '{wf_id}' not found in YAML")
        except Exception as e:
            logger.error(f"Failed to load workflow from YAML. {e}")
            raise

    @property
    def to_dict_exclude_params(self):
        return {"flow": True}

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
        data["flow"] = self.flow.to_dict(include_secure_params=include_secure_params, **kwargs)
        return data

    def to_yaml_file_data(self) -> "WorkflowYamlData":
        """Dump the workflow to a YAML file data.

        Returns:
            WorkflowYamlData: Data for the YAML dump.
        """
        from dynamiq.serializers.loaders.yaml import WorkflowYamlData

        return WorkflowYamlData(
            workflows={self.id: self},
            flows={self.flow.id: self.flow},
            nodes={node.id: node for node in self.flow.nodes},
            connections={},
        )

    def to_yaml_file(self, file_path: str | PathLike | IO[Any]):
        """
        Dump the workflow to a YAML file.

        Args:
            file_path(str | PathLike | IO[Any]): Path to the YAML file.
        """
        from dynamiq.serializers.dumpers.yaml import WorkflowYAMLDumper

        yaml_file_data = self.to_yaml_file_data()

        try:
            WorkflowYAMLDumper.dump(file_path, yaml_file_data)
        except Exception as e:
            logger.error(f"Failed to dump workflow to YAML. {e}")
            raise

    def run(
        self, input_data: Any, config: RunnableConfig = None, **kwargs
    ) -> RunnableResult:
        """Run the workflow with given input data and configuration.

        Args:
            input_data (Any): Input data for the workflow.
            config (RunnableConfig, optional): Configuration for the run. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            RunnableResult: Result of the workflow execution.
        """
        run_id = uuid4()
        logger.info(f"Workflow {self.id}: execution started.")

        # update kwargs with run_id
        merged_kwargs = merge(kwargs, {"run_id": run_id, "wf_run_id": getattr(config, "run_id", None)})
        self.run_on_workflow_start(input_data, config, **merged_kwargs)
        time_start = datetime.now()

        result = self.flow.run(input_data, config, **merge(merged_kwargs, {"parent_run_id": run_id}))
        if result.status == RunnableStatus.SUCCESS:
            self.run_on_workflow_end(result.output, config, **merged_kwargs)
            logger.info(
                f"Workflow {self.id}: execution succeeded in {format_duration(time_start, datetime.now())}."
            )
        else:
            self.run_on_workflow_error(result.output, config, **merged_kwargs)
            logger.error(
                f"Workflow {self.id}: execution failed in {format_duration(time_start, datetime.now())}."
            )

        return RunnableResult(
            status=result.status, input=input_data, output=result.output
        )

    def run_on_workflow_start(self, input_data: Any, config: RunnableConfig = None, **kwargs: Any):
        """Run callbacks on workflow start.

        Args:
            input_data (Any): Input data for the workflow.
            config (RunnableConfig, optional): Configuration for the run. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        if config and config.callbacks:
            for callback in config.callbacks:
                callback.on_workflow_start(self.model_dump(), input_data, **kwargs)

    def run_on_workflow_end(
        self, output: Any, config: RunnableConfig = None, **kwargs: Any
    ):
        """Run callbacks on workflow end.

        Args:
            output (Any): Output data from the workflow.
            config (RunnableConfig, optional): Configuration for the run. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        if config and config.callbacks:
            for callback in config.callbacks:
                callback.on_workflow_end(self.model_dump(), output, **kwargs)

    def run_on_workflow_error(
        self, error: BaseException, config: RunnableConfig = None, **kwargs: Any
    ):
        """Run callbacks on workflow error.

        Args:
            error (BaseException): The error that occurred.
            config (RunnableConfig, optional): Configuration for the run. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        if config and config.callbacks:
            for callback in config.callbacks:
                callback.on_workflow_error(self.model_dump(), error, **kwargs)
