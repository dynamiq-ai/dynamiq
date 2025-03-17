import logging
import os

from dynamiq import Workflow, runnables
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections.managers import get_connection_manager
from dynamiq.serializers.loaders.yaml import WorkflowYAMLLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CAT_FACT_TOKEN = "cat_api_token_12345"  # nosec B105
DOG_FACT_TOKEN = "dog_api_key_67890"  # nosec B105


def run_workflow_yaml_cycle(input_yaml: str, output_yaml: str):
    """Test loading, dumping, and reloading workflow with tool parameters."""
    with get_connection_manager() as cm:

        logger.info("Step 1: Loading original workflow")
        wf_data_original = WorkflowYAMLLoader.load(
            file_path=input_yaml,
            connection_manager=cm,
            init_components=True,
        )
        workflow_1 = Workflow.from_yaml_file_data(file_data=wf_data_original, wf_id="animal-facts-workflow")

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
        workflow_2 = Workflow.from_yaml_file_data(file_data=wf_data_loaded, wf_id="animal-facts-workflow")

        tool_params = {
            "global": {"timeout": 30},  # Apply to all tools
            "by_name": {"CatFactApi": {"headers": {"Authorization": f"Bearer {CAT_FACT_TOKEN}"}}},
            "by_id": {"dog-facts-api-789": {"headers": {"Authorization": f"Bearer {DOG_FACT_TOKEN}"}}},
        }

        test_input = {"input": "Get me a cat fact and a dog fact", "tool_params": tool_params}

        logger.info("Step 4: Testing original workflow")
        tracer_1 = TracingCallbackHandler()
        result_1 = workflow_1.run(input_data=test_input, config=runnables.RunnableConfig(callbacks=[tracer_1]))

        logger.info("Step 5: Testing loaded workflow")
        tracer_2 = TracingCallbackHandler()
        result_2 = workflow_2.run(input_data=test_input, config=runnables.RunnableConfig(callbacks=[tracer_2]))

        return result_1, result_2


if __name__ == "__main__":
    input_yaml = os.path.join(os.path.dirname(__file__), "agent_with_tool_params.yaml")
    output_yaml = "agent_with_tools_dump.yaml"

    try:
        result_1, result_2 = run_workflow_yaml_cycle(input_yaml, output_yaml)

        logger.info("Workflow cycle test completed successfully")
        logger.info(f"Original workflow result status: {result_1.status}")
        logger.info(f"Loaded workflow result status: {result_2.status}")

        if result_1.status == result_2.status:
            logger.info("Both workflows executed with same status")

            if result_1.status == runnables.RunnableStatus.SUCCESS:
                original_content = result_1.output.get("animal-facts-agent", {}).get("output", {}).get("content", "")
                loaded_content = result_2.output.get("animal-facts-agent", {}).get("output", {}).get("content", "")

                logger.info("Original workflow result content (truncated):")
                logger.info(original_content[:200] + "..." if len(original_content) > 200 else original_content)

                logger.info("Loaded workflow result content (truncated):")
                logger.info(loaded_content[:200] + "..." if len(loaded_content) > 200 else loaded_content)
        else:
            logger.warning("Workflows executed with different status")
            logger.warning(f"Original workflow output: {result_1.output}")
            logger.warning(f"Loaded workflow output: {result_2.output}")

    except Exception as e:
        logger.error(f"Workflow cycle test failed: {e}")
