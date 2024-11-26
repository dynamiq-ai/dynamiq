import typing
from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from dynamiq.connections.managers import ConnectionManager
from dynamiq.flows import BaseFlow, Flow
from dynamiq.runnables import Runnable, RunnableConfig, RunnableResult, RunnableStatus
from dynamiq.utils import format_duration, generate_uuid, merge
from dynamiq.utils.logger import logger

if typing.TYPE_CHECKING:
    from dynamiq.loaders.yaml import WorkflowYamlLoaderData


class Workflow(BaseModel, Runnable):
    """Workflow class for managing and running workflows.

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
        from dynamiq.loaders.yaml import WorkflowYAMLLoader

        try:
            wf_data = WorkflowYAMLLoader.load(
                file_path, connection_manager, init_components
            )
        except Exception as e:
            logger.error(f"Failed to load workflow from YAML. {e}")
            raise

        return cls.from_yaml_file_data(wf_data, wf_id)

    @classmethod
    def from_yaml_file_data(
        cls, file_data: "WorkflowYamlLoaderData", wf_id: str = None
    ):
        """Load workflow from YAML file data.

        Args:
            file_data (WorkflowYamlLoaderData): Data loaded from the YAML file.
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
