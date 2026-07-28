import os
import time
import uuid

import pytest

from dynamiq.connections import VertexAI
from dynamiq.nodes.knowledgebases.vertex import CLOUD_PLATFORM_SCOPE, VertexAIRagSearch
from dynamiq.runnables import RunnableStatus

pytestmark = pytest.mark.integration

GCP_ENV_VARS = (
    "GOOGLE_CLOUD_PROJECT_ID",
    "GOOGLE_CLOUD_PRIVATE_KEY",
    "GOOGLE_CLOUD_CLIENT_EMAIL",
    "GOOGLE_CLOUD_TOKEN_URI",
    "VERTEXAI_PROJECT_ID",
    "VERTEXAI_PROJECT_LOCATION",
)

SAMPLE_DOCUMENT = (
    "Dynamiq refund policy: customers can request a full refund within 30 days of purchase. "
    "Refunds are processed by the support department within 5 business days. "
    "After 30 days, purchases are only eligible for account credit."
)
QUERY = "How many days do customers have to request a full refund?"

CORPUS_TIMEOUT_SECONDS = 600
INDEX_TIMEOUT_SECONDS = 600
POLL_INTERVAL_SECONDS = 10


def _gcp_env_available() -> bool:
    return all(os.getenv(variable) for variable in GCP_ENV_VARS)


def _credentials(connection: VertexAI):
    from google.oauth2 import service_account

    info = {
        "type": "service_account",
        "project_id": connection.project_id,
        "private_key_id": connection.private_key_id,
        "private_key": connection.private_key,
        "client_email": connection.client_email,
        "client_id": connection.client_id,
        "auth_uri": connection.auth_uri,
        "token_uri": connection.token_uri,
        "auth_provider_x509_cert_url": connection.auth_provider_x509_cert_url,
        "client_x509_cert_url": connection.client_x509_cert_url,
        "universe_domain": connection.universe_domain,
    }
    info = {key: value for key, value in info.items() if value}
    return service_account.Credentials.from_service_account_info(info, scopes=[CLOUD_PLATFORM_SCOPE])


@pytest.fixture(scope="module")
def rag_corpus_id():
    """Provide a RAG corpus: an existing one from the env, or a bootstrapped temporary corpus.

    Bootstrap mode creates a RagManagedDb-backed corpus, uploads a small text file, waits until
    retrieval returns results, yields the corpus resource name, and deletes the corpus afterwards.
    """
    if not _gcp_env_available():
        pytest.skip("Google Cloud credentials are not set")

    existing_corpus = os.getenv("VERTEXAI_RAG_CORPUS_ID")
    if existing_corpus:
        yield existing_corpus
        return

    from google.api_core.client_options import ClientOptions
    from google.cloud import aiplatform_v1

    connection = VertexAI()
    project = connection.vertex_project_id or connection.project_id
    location = connection.vertex_project_location
    credentials = _credentials(connection)

    data_client = aiplatform_v1.VertexRagDataServiceClient(
        credentials=credentials,
        client_options=ClientOptions(api_endpoint=f"{location}-aiplatform.googleapis.com"),
    )
    parent = f"projects/{project}/locations/{location}"

    operation = data_client.create_rag_corpus(
        parent=parent,
        rag_corpus=aiplatform_v1.RagCorpus(
            display_name=f"dynamiq-e2e-{uuid.uuid4().hex[:8]}",
            description="Temporary corpus created by dynamiq integration tests; safe to delete.",
        ),
    )
    corpus = operation.result(timeout=CORPUS_TIMEOUT_SECONDS)

    try:
        # File upload goes through the SDK's REST upload path; the process-global vertexai.init is
        # acceptable in this single-tenant test process (the node itself never uses it).
        import tempfile

        import vertexai
        from vertexai import rag

        vertexai.init(project=project, location=location, credentials=credentials)
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
            handle.write(SAMPLE_DOCUMENT)
            sample_path = handle.name
        try:
            rag.upload_file(corpus_name=corpus.name, path=sample_path, display_name="refund-policy.txt")
        finally:
            os.unlink(sample_path)

        # Indexing is asynchronous: wait until retrieval through the node returns results.
        probe = VertexAIRagSearch(connection=connection, rag_corpus_id=corpus.name, top_k=3)
        deadline = time.time() + INDEX_TIMEOUT_SECONDS
        while time.time() < deadline:
            result = probe.run(input_data={"query": QUERY})
            if result.status == RunnableStatus.SUCCESS and result.output["documents"]:
                break
            time.sleep(POLL_INTERVAL_SECONDS)
        else:
            raise AssertionError(f"Uploaded file was not retrievable within {INDEX_TIMEOUT_SECONDS}s")

        yield corpus.name
    finally:
        try:
            # force=True: the corpus contains the uploaded RagFile and non-empty corpora
            # cannot be deleted otherwise.
            delete_request = aiplatform_v1.DeleteRagCorpusRequest(name=corpus.name, force=True)
            data_client.delete_rag_corpus(request=delete_request).result(timeout=CORPUS_TIMEOUT_SECONDS)
        except Exception as e:  # noqa: BLE001 - best-effort teardown
            print(f"Failed to delete RAG corpus {corpus.name}: {e}")


@pytest.fixture
def tool(rag_corpus_id):
    return VertexAIRagSearch(connection=VertexAI(), rag_corpus_id=rag_corpus_id, top_k=5)


def test_retrieve_returns_relevant_documents(tool):
    result = tool.run(input_data={"query": QUERY})

    assert result.status == RunnableStatus.SUCCESS
    documents = result.output["documents"]
    assert documents, "expected at least one retrieved document"
    assert "30 days" in result.output["content"]
    top_document = documents[0]
    assert top_document.content
    assert top_document.metadata.get("rag_corpus")


def test_retrieve_respects_top_k(tool):
    result = tool.run(input_data={"query": QUERY, "top_k": 1})

    assert result.status == RunnableStatus.SUCCESS
    assert len(result.output["documents"]) <= 1


@pytest.mark.asyncio
async def test_retrieve_async_returns_documents(tool):
    result = await tool.run_async(input_data={"query": QUERY})

    assert result.status == RunnableStatus.SUCCESS
    assert result.output["documents"]
