import json
import logging
import os

from dotenv import find_dotenv, load_dotenv

from dynamiq import Workflow, runnables
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections.managers import ConnectionManager, get_connection_manager
from dynamiq.serializers.loaders.yaml import WorkflowYAMLLoader
from dynamiq.utils import JsonWorkflowEncoder

logger = logging.getLogger(__name__)
load_dotenv(find_dotenv())


def retrieval_flow(yaml_file_path: str, cm: ConnectionManager):
    wf_data = WorkflowYAMLLoader.load(
        file_path=yaml_file_path,
        connection_manager=cm,
        init_components=True,
    )
    tracing_retrieval_wf = TracingCallbackHandler()
    retrieval_wf = Workflow.from_yaml_file_data(file_data=wf_data, wf_id="retrieval-workflow")
    result = retrieval_wf.run(
        input_data={"query": "How to build an advanced RAG pipeline?"},
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
        for yaml_file_name in ("dag_llm.yaml",):
            dag_yaml_file_path = os.path.join(os.path.dirname(__file__), yaml_file_name)

            result_retrieval, dumped_traces_retrieval_wf = retrieval_flow(
                yaml_file_path=dag_yaml_file_path,
                cm=cm,
            )

            print(result_retrieval)
            answer = result_retrieval.output.get("openai-1").get("output").get("answer")

            logger.info(f"Answer:\n {answer}")
