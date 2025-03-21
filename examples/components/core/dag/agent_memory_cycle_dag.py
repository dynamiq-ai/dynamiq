import logging
import os

from dynamiq import Workflow, runnables
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections.managers import get_connection_manager
from dynamiq.serializers.loaders.yaml import WorkflowYAMLLoader

logger = logging.getLogger(__name__)


def run_workflow_yaml_cycle(input_yaml: str, output_yaml: str):
    """Test loading, dumping, and reloading workflow."""
    with get_connection_manager() as cm:

        logger.info("Step 1: Loading original workflow")
        wf_data_original = WorkflowYAMLLoader.load(
            file_path=input_yaml,
            connection_manager=cm,
            init_components=True,
        )
        workflow_1 = Workflow.from_yaml_file_data(file_data=wf_data_original, wf_id="memory-agent-workflow")

        logger.info("Step 2: Dumping workflow to new YAML")
        try:
            workflow_1.to_yaml_file(output_yaml)
        except Exception as e:
            logger.error(f"Failed to dump workflow: {e}")
            raise

        logger.info("Step 3: Loading workflow from dumped YAML")
        wf_data_loaded = WorkflowYAMLLoader.load(
            file_path=output_yaml,
            connection_manager=cm,
            init_components=True,
        )
        workflow_2 = Workflow.from_yaml_file_data(file_data=wf_data_loaded, wf_id="memory-agent-workflow")

        test_input = {"input": "Hello! Can you help me with information about artificial intelligence?"}

        logger.info("Step 4: Testing original workflow")
        result_1 = workflow_1.run(
            input_data=test_input, config=runnables.RunnableConfig(callbacks=[TracingCallbackHandler()])
        )

        logger.info("Step 5: Testing loaded workflow")
        result_2 = workflow_2.run(
            input_data=test_input, config=runnables.RunnableConfig(callbacks=[TracingCallbackHandler()])
        )

        return result_1, result_2


if __name__ == "__main__":
    input_yaml = os.path.join(os.path.dirname(__file__), "agent_memory_dag.yaml")
    output_yaml = "agent_memory_dag_dump.yaml"

    try:
        result_1, result_2 = run_workflow_yaml_cycle(input_yaml, output_yaml)

        logger.info("Workflow cycle test completed successfully")
        logger.info(f"Original workflow result: {result_1.output}")
        logger.info(f"Loaded workflow result: {result_2.output}")

        if result_1.status == result_2.status:
            logger.info("Both workflows executed with same status")
        else:
            logger.warning("Workflows executed with different status")

    except Exception as e:
        logger.error(f"Workflow cycle test failed: {e}")
