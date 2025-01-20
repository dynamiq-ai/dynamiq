import json
import logging
import os

from dynamiq import Workflow, runnables
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections.managers import ConnectionManager, get_connection_manager
from dynamiq.serializers.loaders.yaml import WorkflowYAMLLoader
from dynamiq.utils import JsonWorkflowEncoder

logger = logging.getLogger(__name__)


def get_current_time(location: str) -> str:
    """Get the current time in a given location."""
    location_times = {
        "tokyo": {"location": "Tokyo", "current_time": "14:00"},
        "san francisco": {"location": "San Francisco", "current_time": "22:00"},
        "paris": {"location": "Paris", "current_time": "06:00"},
    }

    normalized_location = location.lower()

    if normalized_location in location_times:
        response_data = location_times[normalized_location]
    else:
        response_data = {"location": location, "current_time": "unknown"}

    return json.dumps(response_data)


def retrieval_flow(yaml_file_path: str, cm: ConnectionManager):
    wf_data = WorkflowYAMLLoader.load(
        file_path=yaml_file_path,
        connection_manager=cm,
        init_components=True,
    )
    tracing_retrieval_wf = TracingCallbackHandler()
    retrieval_wf = Workflow.from_yaml_file_data(file_data=wf_data, wf_id="retrieval-workflow")
    result = retrieval_wf.run(
        input_data={},
        config=runnables.RunnableConfig(callbacks=[tracing_retrieval_wf]),
    )
    dumped_traces_retrieval_wf = json.dumps(
        {"runs": [run.to_dict() for run in tracing_retrieval_wf.runs.values()]},
        cls=JsonWorkflowEncoder,
    )
    return result, dumped_traces_retrieval_wf


if __name__ == "__main__":
    data_folder_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    with get_connection_manager() as cm:
        for yaml_file_name in ("dag_llm_tools.yaml",):
            dag_yaml_file_path = os.path.join(os.path.dirname(__file__), yaml_file_name)

            result_retrieval, dumped_traces_retrieval_wf = retrieval_flow(
                yaml_file_path=dag_yaml_file_path,
                cm=cm,
            )

            tool_calls = result_retrieval.output.get("openai-1").get("output").get("tool_calls")

            if tool_calls:
                available_functions = {
                    "get_current_time": get_current_time,
                }

                for tool_call in list(tool_calls.values()):
                    logger.info(f"\nExecuting tool call\n{tool_call}")
                    function_name = tool_call["function"]["name"]
                    function_to_call = available_functions[function_name]
                    function_args = json.loads(tool_call["function"]["arguments"])
                    function_response = function_to_call(
                        location=function_args.get("location"),
                    )
                    logger.info(f"Result from tool call\n{function_response}\n")
            logger.info(f"Workflow result:{result_retrieval}")
