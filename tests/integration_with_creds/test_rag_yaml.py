import os

import pytest

from dynamiq import ROOT_PATH
from dynamiq.connections.managers import get_connection_manager
from dynamiq.runnables import RunnableStatus
from examples.components.rag.vector_stores.dag.dag_yaml import indexing_flow, retrieval_flow


@pytest.fixture
def rag_examples_folder():
    return os.path.join(os.path.dirname(ROOT_PATH), "examples", "components", "rag", "vector_stores", "dag")


@pytest.fixture
def rag_data_path():
    return os.path.join(os.path.dirname(ROOT_PATH), "examples", "components", "data")


@pytest.mark.parametrize("rag_yaml_file_name", ["dag_html_pinecone.yaml"])
def test_indexing_flow(rag_examples_folder, rag_data_path, rag_yaml_file_name):
    with get_connection_manager() as cm:
        result, dumped_tracing = indexing_flow(
            yaml_file_path=os.path.join(rag_examples_folder, rag_yaml_file_name),
            data_folder_path=rag_data_path,
            cm=cm,
            extensions=[".html"],
            init_components=False,  # Lazy init to avoid initializing retriever before index exists
        )
    assert result.status == RunnableStatus.SUCCESS
    assert dumped_tracing


@pytest.mark.parametrize("rag_yaml_file_name", ["dag_pinecone.yaml"])
def test_retrival_flow(rag_examples_folder, rag_data_path, rag_yaml_file_name):
    with get_connection_manager() as cm:
        result, dumped_tracing = retrieval_flow(
            yaml_file_path=os.path.join(rag_examples_folder, rag_yaml_file_name),
            cm=cm,
        )
    assert result.status == RunnableStatus.SUCCESS
    assert dumped_tracing
