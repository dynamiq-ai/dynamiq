import json
import logging
import os

from dotenv import find_dotenv, load_dotenv

from dynamiq import Workflow, runnables
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections.managers import ConnectionManager, get_connection_manager
from dynamiq.serializers.loaders.yaml import WorkflowYAMLLoader
from dynamiq.utils import JsonWorkflowEncoder
from examples.components.rag.vector_stores.utils import list_data_folder_paths, read_bytes_io_files

logger = logging.getLogger(__name__)
load_dotenv(find_dotenv())


def indexing_flow(
    yaml_file_path: str,
    data_folder_path: str,
    cm: ConnectionManager,
    extensions: list[str] | None = None,
):
    wf_data = WorkflowYAMLLoader.load(
        file_path=yaml_file_path,
        connection_manager=cm,
        init_components=True,
    )
    tracing_indexing_wf = TracingCallbackHandler()
    indexing_wf = Workflow.from_yaml_file_data(file_data=wf_data, wf_id="indexing-workflow")

    file_paths = list_data_folder_paths(data_folder_path, extensions=extensions)
    input_data = read_bytes_io_files(file_paths)

    result = indexing_wf.run(
        input_data=input_data,
        config=runnables.RunnableConfig(callbacks=[tracing_indexing_wf]),
    )
    dumped_traces_indexing_wf = json.dumps(
        {"runs": [run.to_dict() for run in tracing_indexing_wf.runs.values()]},
        cls=JsonWorkflowEncoder,
    )
    return result, dumped_traces_indexing_wf


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
        for yaml_file_name in (
            "dag_weaviate_tenant.yaml",
            "dag_html_pinecone.yaml",
            "dag_qdrant.yaml",
            "dag_chroma.yaml",
            "dag_pinecone.yaml",
            # "dag_weaviate_custom.yaml",
            "dag_weaviate.yaml",
            "dag_pgvector.yaml",
            "dag_elasticsearch.yaml",
        ):
            dag_yaml_file_path = os.path.join(os.path.dirname(__file__), yaml_file_name)
            result_indexing, dumped_traces_indexing_wf = indexing_flow(
                yaml_file_path=dag_yaml_file_path,
                data_folder_path=data_folder_path,
                cm=cm,
            )

            result_retrieval, dumped_traces_retrieval_wf = retrieval_flow(
                yaml_file_path=dag_yaml_file_path,
                cm=cm,
            )

            answer = result_retrieval.output.get("openai-1").get("output").get("answer")

            logger.info(f"Retrival output (Answer):\n {answer}")
