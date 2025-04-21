import uuid

from dynamiq import Workflow, connections
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes.embedders import VertexAIDocumentEmbedder, VertexAITextEmbedder
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus
from dynamiq.types import Document


def test_workflow_with_vertexai_text_embedder(mock_embedding_executor):
    # Create connection with mock credentials
    connection = connections.VertexAI(
        id=str(uuid.uuid4()),
        project_id="mock_project",
        private_key_id="mock_key_id",
        private_key="mock_private_key",
        client_email="mock_client_email",
        client_id="mock_client_id",
        auth_uri="https://mock.auth.uri",
        token_uri="https://mock.token.uri",
        auth_provider_x509_cert_url="https://mock.cert.url",
        client_x509_cert_url="https://mock.client.cert.url",
        universe_domain="mock_domain",
        vertex_project_id="mock_vertex_project",
        vertex_project_location="mock_location",
    )
    model = "vertex_ai/text-embedding-005"
    wf = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(nodes=[VertexAITextEmbedder(name="VertexAITextEmbedder", connection=connection, model=model)]),
    )
    input_data = {"query": "I love pizza!"}
    response = wf.run(
        input_data=input_data,
        config=RunnableConfig(callbacks=[TracingCallbackHandler()]),
    )

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output={"query": "I love pizza!", "embedding": [0]},
    ).to_dict()
    expected_output = {wf.flow.nodes[0].id: expected_result}

    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output=expected_output,
    )
    mock_embedding_executor.assert_called_once_with(
        task_type="RETRIEVAL_QUERY",
        input=[input_data["query"]],
        model=model,
        **connection.conn_params,
    )


def test_workflow_with_vertexai_document_embedder(mock_embedding_executor):
    # Create connection with mock credentials
    connection = connections.VertexAI(
        id=str(uuid.uuid4()),
        project_id="mock_project",
        private_key_id="mock_key_id",
        private_key="mock_private_key",
        client_email="mock_client_email",
        client_id="mock_client_id",
        auth_uri="https://mock.auth.uri",
        token_uri="https://mock.token.uri",
        auth_provider_x509_cert_url="https://mock.cert.url",
        client_x509_cert_url="https://mock.client.cert.url",
        universe_domain="mock_domain",
        vertex_project_id="mock_vertex_project",
        vertex_project_location="mock_location",
    )
    model = "vertex_ai/text-embedding-005"
    wf = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(
            nodes=[VertexAIDocumentEmbedder(name="VertexAIDocumentEmbedder", connection=connection, model=model)]
        ),
    )
    document = [Document(content="I love pizza!")]
    input_data = {"documents": document}
    response = wf.run(
        input_data=input_data,
        config=RunnableConfig(callbacks=[TracingCallbackHandler()]),
    )

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output={
            **input_data,
            "meta": {
                "model": model,
                "usage": {"usage": {"prompt_tokens": 6, "completion_tokens": 0, "total_tokens": 6}},
            },
        },
    ).to_dict()
    expected_output = {wf.flow.nodes[0].id: expected_result}

    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output=expected_output,
    )
    mock_embedding_executor.assert_called_once_with(
        task_type="RETRIEVAL_DOCUMENT",
        input=[document[0].content],
        model=model,
        **connection.conn_params,
    )
