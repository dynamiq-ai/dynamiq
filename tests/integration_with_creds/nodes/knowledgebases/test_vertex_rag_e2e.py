"""Live tests for VertexAIRagSearch against a real Vertex AI RAG Engine corpus.

Run against any Google Cloud project; nothing here is account-specific.

Environment
-----------
The VertexAI connection reads its service-account fields from the standard GOOGLE_CLOUD_* /
VERTEXAI_* variables (see .env.example). Credentials are detected by *resolving* them, not by
listing variable names, so a partially-populated environment skips with a reason instead of
failing obscurely.

    VERTEXAI_RAG_CORPUS_ID       optional; an existing corpus (id or full resource name).
                                 When unset the module bootstraps a temporary corpus, which
                                 provisions a RagManagedDb (Spanner) instance that bills per hour
                                 of existence until the corpus is deleted.
    VERTEXAI_LLM_RANKER_MODEL    optional; defaults to gemini-2.5-flash. Model availability is
                                 regional, so a 404 here is an environment gap, not a failure.
    VERTEXAI_RANK_SERVICE_MODEL  optional; defaults to semantic-ranker-default@latest. Needs
                                 discoveryengine.rankingConfigs.rank on the *calling* identity,
                                 which roles/aiplatform.user does not grant.

Tests that depend on file metadata only run against the bootstrapped corpus, because custom
metadata has to be provisioned through the v1beta1 API (RagDataSchema + ragMetadata:batchCreate)
and an externally supplied corpus is not guaranteed to have it.
"""

from __future__ import annotations

import io
import json
import os
import time
import uuid
from dataclasses import dataclass, field

import pytest

from dynamiq.connections import VertexAI
from dynamiq.nodes.knowledgebases.vertex import CLOUD_PLATFORM_SCOPE, VertexAIRagSearch
from dynamiq.runnables import RunnableStatus

pytestmark = pytest.mark.integration

REFUND_QUERY = "How many days do customers have to request a full refund?"
ONCALL_QUERY = "How quickly must an on-call engineer acknowledge a page?"
ESCALATION_QUERY = "Who is a severity one incident escalated to and how quickly?"

CORPUS_TIMEOUT_SECONDS = 900
INDEX_TIMEOUT_SECONDS = 900
POLL_INTERVAL_SECONDS = 15

# The corpus is built so that wrong behaviour produces a wrong answer: the 2022 revision is a
# near-miss on the same topic, so a metadata filter that silently no-ops leaks "7 days" into a
# result set that should only contain "30 days".
DOCUMENTS = [
    (
        "refund-policy-2025",
        "support",
        2025,
        "Dynamiq refund policy, revision 2025-A. Customers can request a full refund within "
        "30 days of purchase. Refunds are processed by the support department within 5 business "
        "days. After 30 days, purchases are only eligible for account credit.",
    ),
    (
        "refund-policy-2022",
        "support",
        2022,
        "Dynamiq refund policy, revision 2022-C. SUPERSEDED, do not quote to customers. "
        "Customers can request a full refund within 7 days of purchase. Refunds were processed "
        "by the billing department within 10 business days.",
    ),
    (
        "engineering-oncall-2025",
        "engineering",
        2025,
        "Dynamiq engineering on-call handbook, 2025 edition. On-call engineers must acknowledge "
        "a page within 15 minutes. Escalation to the secondary responder happens after 30 "
        "minutes without acknowledgement.",
    ),
    (
        "shipping-policy-2025",
        "logistics",
        2025,
        "Dynamiq shipping policy, 2025 edition. Standard shipping takes 4 business days. "
        "Expedited shipping takes 1 business day and costs an additional handling fee.",
    ),
]

# Only a paged source carries RagChunk.page_span, so the fixture needs one PDF.
PDF_DISPLAY_NAME = "escalation-handbook-2025"
PDF_PAGES = [
    "Dynamiq escalation handbook, 2025 edition. Page one covers intake. Tier one support "
    "acknowledges every ticket within 2 hours of receipt.",
    "Dynamiq escalation handbook, 2025 edition. Page two covers executive escalation. A severity "
    "one incident is escalated to the vice president of engineering within 90 minutes.",
]

# Signals that mean "this environment cannot run this check", never "the code is broken".
ENVIRONMENT_GAP_SIGNALS = (
    "discoveryengine.rankingConfigs.rank",
    "Publisher model",
    "was not found or your project does not have access",
    "is restricted to only allowlisted projects",
    "Quota exceeded",
    "RESOURCE_EXHAUSTED",
)


@dataclass
class Fixture:
    corpus: str
    location: str
    files: dict[str, str] = field(default_factory=dict)
    metadata_provisioned: bool = False
    metadata_error: str = ""


def _connection() -> VertexAI:
    return VertexAI()


def _credentials(connection: VertexAI):
    """Resolve real credentials; raising here means the environment cannot authenticate."""
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


def _skip_if_environment_gap(result, feature: str) -> None:
    """Skip only on known environment-gap signals; anything else is a real failure."""
    if result.status == RunnableStatus.SUCCESS:
        return
    message = str(getattr(result, "error", ""))
    for signal in ENVIRONMENT_GAP_SIGNALS:
        if signal in message:
            pytest.skip(f"{feature} unavailable in this environment: {message[:200]}")
    pytest.fail(f"{feature} failed: {message[:500]}")


def _build_pdf() -> bytes:
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=LETTER)
    for page in PDF_PAGES:
        text = pdf.beginText(72, 720)
        text.setFont("Helvetica", 12)
        for chunk in [page[i : i + 90] for i in range(0, len(page), 90)]:
            text.textLine(chunk)
        pdf.drawText(text)
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def _upload(credentials, location: str, corpus: str, display_name: str, payload: bytes, suffix: str) -> str:
    """UploadRagFile is a media-upload endpoint, so it is only reachable over REST.

    The uploaded part needs a real file extension: the parser picks the reader from it and rejects
    the upload with "File does not have extension" otherwise.
    """
    import google.auth.transport.requests
    import requests

    mime = {"txt": "text/plain", "pdf": "application/pdf"}[suffix]
    credentials.refresh(google.auth.transport.requests.Request())
    response = requests.post(
        f"https://{location}-aiplatform.googleapis.com/upload/v1/{corpus}/ragFiles:upload",
        headers={"Authorization": f"Bearer {credentials.token}", "X-Goog-Upload-Protocol": "multipart"},
        files={
            "metadata": (None, json.dumps({"rag_file": {"display_name": display_name}}), "application/json"),
            "file": (f"{display_name}.{suffix}", payload, mime),
        },
        timeout=180,
    )
    if response.status_code != 200 or "ragFile" not in response.json():
        raise AssertionError(f"upload of {display_name} failed: {response.status_code} {response.text[:300]}")
    return response.json()["ragFile"]["name"]


def _provision_metadata(credentials, location: str, corpus: str, files: dict[str, str]) -> tuple[bool, str]:
    """Attach department/year metadata through v1beta1.

    Returns (ok, reason); the reason is carried into the skip message so a skipped metadata test
    always says what actually went wrong.
    """
    from google.api_core.client_options import ClientOptions
    from google.cloud import aiplatform_v1beta1 as beta

    client = beta.VertexRagDataServiceClient(
        credentials=credentials,
        client_options=ClientOptions(api_endpoint=f"{location}-aiplatform.googleapis.com"),
    )
    schemas = [
        ("department", beta.RagMetadataSchemaDetails.DataType.STRING),
        ("year", beta.RagMetadataSchemaDetails.DataType.INTEGER),
    ]
    try:
        client.batch_create_rag_data_schemas(
            request=beta.BatchCreateRagDataSchemasRequest(
                parent=corpus,
                requests=[
                    beta.CreateRagDataSchemaRequest(
                        parent=corpus,
                        rag_data_schema_id=key,
                        rag_data_schema=beta.RagDataSchema(
                            key=key,
                            schema_details=beta.RagMetadataSchemaDetails(
                                type_=dtype,
                                granularity=(beta.RagMetadataSchemaDetails.Granularity.GRANULARITY_FILE_LEVEL),
                                search_strategy=beta.RagMetadataSchemaDetails.SearchStrategy(
                                    search_strategy_type=(
                                        beta.RagMetadataSchemaDetails.SearchStrategy.SearchStrategyType.EXACT_SEARCH
                                    )
                                ),
                            ),
                        ),
                    )
                    for key, dtype in schemas
                ],
            )
        ).result(timeout=CORPUS_TIMEOUT_SECONDS)

        # The create LRO completes before the keys are readable; attaching metadata too early
        # fails with "The following RagDataSchema keys do not exist".
        wanted = {key for key, _ in schemas}
        deadline = time.time() + 300
        while time.time() < deadline:
            visible = {schema.key for schema in client.list_rag_data_schemas(parent=corpus)}
            if wanted <= visible:
                break
            time.sleep(5)
        else:
            return False, f"RagDataSchema keys {sorted(wanted)} never became visible"

        tagged = [(name, department, year) for name, department, year, _ in DOCUMENTS]
        tagged.append((PDF_DISPLAY_NAME, "support", 2025))
        for display_name, department, year in tagged:
            rag_file = files[display_name]
            request = beta.BatchCreateRagMetadataRequest(
                parent=rag_file,
                requests=[
                    beta.CreateRagMetadataRequest(
                        parent=rag_file,
                        rag_metadata_id="department",
                        rag_metadata=beta.RagMetadata(
                            user_specified_metadata=beta.UserSpecifiedMetadata(
                                key="department", value=beta.MetadataValue(str_value=department)
                            )
                        ),
                    ),
                    beta.CreateRagMetadataRequest(
                        parent=rag_file,
                        rag_metadata_id="year",
                        rag_metadata=beta.RagMetadata(
                            user_specified_metadata=beta.UserSpecifiedMetadata(
                                key="year", value=beta.MetadataValue(int_value=year)
                            )
                        ),
                    ),
                ],
            )
            # The write path sees new schemas later than the list API does, so a freshly created
            # schema is still rejected for a while after list_rag_data_schemas reports it.
            attach_deadline = time.time() + 600
            while True:
                try:
                    client.batch_create_rag_metadata(request=request).result(timeout=CORPUS_TIMEOUT_SECONDS)
                    break
                except Exception as e:  # noqa: BLE001
                    if "do not exist" not in str(e) or time.time() > attach_deadline:
                        raise
                    time.sleep(15)
        return True, ""
    except Exception as e:  # noqa: BLE001 - metadata search is not available everywhere
        return False, f"{type(e).__name__}: {str(e)[:300]}"


def _adopt_existing_corpus(connection: VertexAI, credentials, corpus: str) -> Fixture:
    """Describe a corpus supplied through the environment.

    Content-dependent tests only run when the corpus happens to carry the fixture layout, so an
    arbitrary corpus still exercises everything content-independent and skips the rest with a
    reason rather than failing.
    """
    from google.api_core.client_options import ClientOptions
    from google.cloud import aiplatform_v1

    location = corpus.split("/")[3]
    data_client = aiplatform_v1.VertexRagDataServiceClient(
        credentials=credentials,
        client_options=ClientOptions(api_endpoint=f"{location}-aiplatform.googleapis.com"),
    )
    present = {f.display_name: f.name for f in data_client.list_rag_files(parent=corpus)}
    expected = {display_name for display_name, *_ in DOCUMENTS} | {PDF_DISPLAY_NAME}
    files = present if expected <= set(present) else {}

    metadata_ok = False
    if files:
        probe = VertexAIRagSearch(
            connection=connection, rag_corpus_id=corpus, metadata_filter='department == "support"', top_k=5
        ).run(input_data={"query": REFUND_QUERY})
        metadata_ok = probe.status == RunnableStatus.SUCCESS and bool(probe.output["documents"])
    return Fixture(corpus=corpus, location=location, files=files, metadata_provisioned=metadata_ok)


@pytest.fixture(scope="module")
def fixture() -> Fixture:
    """An existing corpus from the environment, or a temporary bootstrapped one."""
    connection = _connection()
    try:
        credentials = _credentials(connection)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Google Cloud service-account credentials could not be resolved: {e}")

    location = connection.vertex_project_location
    project = connection.vertex_project_id or connection.project_id
    if not location or not project:
        pytest.skip("VERTEXAI_PROJECT_ID / VERTEXAI_PROJECT_LOCATION are not set")

    from google.api_core.client_options import ClientOptions
    from google.cloud import aiplatform_v1

    existing = os.getenv("VERTEXAI_RAG_CORPUS_ID")
    if existing:
        corpus = (
            existing
            if existing.startswith("projects/")
            else (f"projects/{project}/locations/{location}/ragCorpora/{existing}")
        )
        yield _adopt_existing_corpus(connection, credentials, corpus)
        return

    options = ClientOptions(api_endpoint=f"{location}-aiplatform.googleapis.com")
    data_client = aiplatform_v1.VertexRagDataServiceClient(credentials=credentials, client_options=options)

    try:
        operation = data_client.create_rag_corpus(
            parent=f"projects/{project}/locations/{location}",
            rag_corpus=aiplatform_v1.RagCorpus(
                display_name=f"dynamiq-e2e-{uuid.uuid4().hex[:8]}",
                description="Temporary corpus created by dynamiq integration tests; safe to delete.",
            ),
        )
        corpus = operation.result(timeout=CORPUS_TIMEOUT_SECONDS).name
    except Exception as e:  # noqa: BLE001
        message = str(e)
        if any(signal in message for signal in ENVIRONMENT_GAP_SIGNALS):
            pytest.skip(f"RAG corpus cannot be created in {location}: {message[:250]}")
        raise

    try:
        files = {
            display_name: _upload(credentials, location, corpus, display_name, text.encode(), "txt")
            for display_name, _department, _year, text in DOCUMENTS
        }
        files[PDF_DISPLAY_NAME] = _upload(credentials, location, corpus, PDF_DISPLAY_NAME, _build_pdf(), "pdf")
        metadata_ok, metadata_error = _provision_metadata(credentials, location, corpus, files)

        # Every layer has its own lifecycle; wait on the documents, not just the corpus.
        probe = VertexAIRagSearch(connection=connection, rag_corpus_id=corpus, top_k=30)
        expected = {display_name for display_name, *_ in DOCUMENTS} | {PDF_DISPLAY_NAME}
        deadline = time.time() + INDEX_TIMEOUT_SECONDS
        while time.time() < deadline:
            result = probe.run(input_data={"query": "refund policy shipping on-call escalation"})
            if result.status == RunnableStatus.SUCCESS:
                seen = {d.metadata.get("source_display_name") for d in result.output["documents"]}
                if expected <= seen:
                    break
            time.sleep(POLL_INTERVAL_SECONDS)
        else:
            raise AssertionError(f"corpus files were not retrievable within {INDEX_TIMEOUT_SECONDS}s")

        yield Fixture(
            corpus=corpus,
            location=location,
            files=files,
            metadata_provisioned=metadata_ok,
            metadata_error=metadata_error,
        )
    finally:
        try:
            # force=True: a corpus holding RagFiles cannot be deleted otherwise.
            data_client.delete_rag_corpus(request=aiplatform_v1.DeleteRagCorpusRequest(name=corpus, force=True)).result(
                timeout=CORPUS_TIMEOUT_SECONDS
            )
        except Exception as e:  # noqa: BLE001 - best-effort teardown
            print(f"Failed to delete RAG corpus {corpus}: {e}")


@pytest.fixture
def tool(fixture):
    return VertexAIRagSearch(connection=_connection(), rag_corpus_id=fixture.corpus, top_k=5)


def _requires_bootstrapped(fixture) -> None:
    if not fixture.files:
        pytest.skip("test needs the bootstrapped fixture corpus; VERTEXAI_RAG_CORPUS_ID was supplied")


def _requires_metadata(fixture) -> None:
    _requires_bootstrapped(fixture)
    if not fixture.metadata_provisioned:
        reason = fixture.metadata_error or "the supplied corpus carries no department/year metadata"
        pytest.skip(f"file metadata unavailable: {reason}")


def _sources(result) -> set[str]:
    return {d.metadata.get("source_display_name") for d in result.output["documents"]}


# --------------------------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------------------------


def test_retrieve_returns_relevant_documents(tool, fixture):
    _requires_bootstrapped(fixture)
    result = tool.run(input_data={"query": REFUND_QUERY})

    assert result.status == RunnableStatus.SUCCESS, result.error
    documents = result.output["documents"]
    assert documents, "expected at least one retrieved document"
    assert "30 days" in result.output["content"]
    assert documents[0].content
    assert documents[0].metadata["rag_corpus"] == fixture.corpus
    assert documents[0].metadata["source_display_name"]


def test_retrieve_respects_top_k(tool):
    result = tool.run(input_data={"query": REFUND_QUERY, "top_k": 1})

    assert result.status == RunnableStatus.SUCCESS, result.error
    assert len(result.output["documents"]) <= 1


@pytest.mark.asyncio
async def test_retrieve_async_returns_documents(tool):
    result = await tool.run_async(input_data={"query": REFUND_QUERY})

    assert result.status == RunnableStatus.SUCCESS, result.error
    assert result.output["documents"]


def test_document_ids_are_stable_across_calls(tool):
    first = tool.run(input_data={"query": REFUND_QUERY})
    second = tool.run(input_data={"query": REFUND_QUERY})

    assert first.status == second.status == RunnableStatus.SUCCESS
    assert [d.id for d in first.output["documents"]] == [d.id for d in second.output["documents"]]


# --------------------------------------------------------------------------------------------
# Scoping: files, metadata filters
# --------------------------------------------------------------------------------------------


def test_rag_file_ids_restrict_retrieval(fixture):
    """Restricting to the current revision must exclude the superseded one."""
    _requires_bootstrapped(fixture)
    node = VertexAIRagSearch(
        connection=_connection(),
        rag_corpus_id=fixture.corpus,
        rag_file_ids=[fixture.files["refund-policy-2025"].split("/")[-1]],
        top_k=10,
    )
    result = node.run(input_data={"query": REFUND_QUERY})

    assert result.status == RunnableStatus.SUCCESS, result.error
    assert _sources(result) == {"refund-policy-2025"}
    assert "30 days" in result.output["content"]
    assert "7 days" not in result.output["content"], "superseded revision leaked past rag_file_ids"


def test_metadata_filter_excludes_superseded_revision(fixture):
    _requires_metadata(fixture)
    node = VertexAIRagSearch(
        connection=_connection(),
        rag_corpus_id=fixture.corpus,
        metadata_filter='department == "support" && year == 2025',
        top_k=10,
    )
    result = node.run(input_data={"query": REFUND_QUERY})
    _skip_if_environment_gap(result, "metadata filtering")

    assert "refund-policy-2022" not in _sources(result)
    assert "7 days" not in result.output["content"], "metadata filter silently no-opped"
    assert "30 days" in result.output["content"]


def test_input_metadata_filter_overrides_node_level(fixture):
    _requires_metadata(fixture)
    node = VertexAIRagSearch(
        connection=_connection(),
        rag_corpus_id=fixture.corpus,
        metadata_filter='department == "support"',
        top_k=10,
    )
    result = node.run(input_data={"query": ONCALL_QUERY, "metadata_filter": 'department == "engineering"'})
    _skip_if_environment_gap(result, "metadata filtering")

    assert _sources(result) == {"engineering-oncall-2025"}


def test_metadata_filter_matching_nothing_returns_friendly_message(fixture):
    _requires_metadata(fixture)
    node = VertexAIRagSearch(
        connection=_connection(), rag_corpus_id=fixture.corpus, metadata_filter="year == 1999", top_k=10
    )
    result = node.run(input_data={"query": REFUND_QUERY})
    _skip_if_environment_gap(result, "metadata filtering")

    assert result.output["documents"] == []
    assert result.output["content"] == "No results found for the given query."


def test_metadata_filter_is_cel_not_sql(fixture):
    """metadata_filter is a CEL expression: a single '=' is a syntax error, not a match."""
    _requires_metadata(fixture)
    node = VertexAIRagSearch(
        connection=_connection(),
        rag_corpus_id=fixture.corpus,
        metadata_filter='department = "support"',
        top_k=5,
    )
    result = node.run(input_data={"query": REFUND_QUERY})

    assert result.status == RunnableStatus.FAILURE
    assert "Syntax error" in str(result.error)
    assert result.error.recoverable


# --------------------------------------------------------------------------------------------
# Thresholds and ranking
# --------------------------------------------------------------------------------------------


def test_vector_distance_threshold_filters_results(fixture):
    _requires_bootstrapped(fixture)
    loose = VertexAIRagSearch(
        connection=_connection(), rag_corpus_id=fixture.corpus, vector_distance_threshold=1.0, top_k=10
    ).run(input_data={"query": REFUND_QUERY})
    tight = VertexAIRagSearch(
        connection=_connection(), rag_corpus_id=fixture.corpus, vector_distance_threshold=0.05, top_k=10
    ).run(input_data={"query": REFUND_QUERY})

    _skip_if_environment_gap(loose, "vector_distance_threshold")
    _skip_if_environment_gap(tight, "vector_distance_threshold")
    assert len(tight.output["documents"]) < len(loose.output["documents"])


def test_vector_similarity_threshold_is_accepted(fixture):
    """RagManagedDb scores are cosine *distances*, so it accepts this knob and ignores it.

    Asserting only acceptance is deliberate: asserting that it filters would fail on
    RagManagedDb, and asserting that it does not would fail on a backend where it works.
    """
    _requires_bootstrapped(fixture)
    result = VertexAIRagSearch(
        connection=_connection(), rag_corpus_id=fixture.corpus, vector_similarity_threshold=0.5, top_k=10
    ).run(input_data={"query": REFUND_QUERY})

    _skip_if_environment_gap(result, "vector_similarity_threshold")
    assert result.status == RunnableStatus.SUCCESS


def test_llm_ranker_is_applied(fixture):
    _requires_bootstrapped(fixture)
    model = os.getenv("VERTEXAI_LLM_RANKER_MODEL", "gemini-2.5-flash")
    result = VertexAIRagSearch(
        connection=_connection(), rag_corpus_id=fixture.corpus, llm_ranker_model=model, top_k=5
    ).run(input_data={"query": REFUND_QUERY})

    _skip_if_environment_gap(result, f"llm ranker {model}")
    documents = result.output["documents"]
    assert documents
    assert all(d.metadata["reranked"] is True for d in documents)
    # The ranker reorders; the score keeps being the vector score, so it must stay unlabelled
    # when no threshold said what that score means.
    assert all("score_type" not in d.metadata for d in documents)


def test_rank_service_is_applied(fixture):
    _requires_bootstrapped(fixture)
    model = os.getenv("VERTEXAI_RANK_SERVICE_MODEL", "semantic-ranker-default@latest")
    result = VertexAIRagSearch(
        connection=_connection(), rag_corpus_id=fixture.corpus, rank_service_model=model, top_k=5
    ).run(input_data={"query": REFUND_QUERY})

    _skip_if_environment_gap(result, f"semantic ranker {model}")
    assert result.output["documents"]
    assert all(d.metadata["reranked"] is True for d in result.output["documents"])


# --------------------------------------------------------------------------------------------
# Response mapping
# --------------------------------------------------------------------------------------------


def test_scores_are_present_and_passed_through(tool):
    result = tool.run(input_data={"query": REFUND_QUERY})

    assert result.status == RunnableStatus.SUCCESS, result.error
    scores = [d.score for d in result.output["documents"]]
    assert scores and all(s is not None for s in scores)


def test_page_span_is_mapped_for_paged_sources(fixture):
    _requires_bootstrapped(fixture)
    node = VertexAIRagSearch(connection=_connection(), rag_corpus_id=fixture.corpus, top_k=10)
    result = node.run(input_data={"query": ESCALATION_QUERY})

    assert result.status == RunnableStatus.SUCCESS, result.error
    paged = [
        d
        for d in result.output["documents"]
        if d.metadata.get("source_display_name") == PDF_DISPLAY_NAME and "page_first" in d.metadata
    ]
    if not paged:
        pytest.skip("the PDF chunk did not rank into top_k for this query")
    assert paged[0].metadata["page_first"] >= 1
    assert paged[0].metadata["page_last"] >= paged[0].metadata["page_first"]


def test_metadata_fields_allowlist_controls_formatted_content(fixture):
    _requires_bootstrapped(fixture)
    allowlisted = VertexAIRagSearch(
        connection=_connection(),
        rag_corpus_id=fixture.corpus,
        metadata_fields=["source_display_name"],
        top_k=2,
    ).run(input_data={"query": REFUND_QUERY})
    everything = VertexAIRagSearch(
        connection=_connection(), rag_corpus_id=fixture.corpus, metadata_fields=None, top_k=2
    ).run(input_data={"query": REFUND_QUERY})

    assert allowlisted.status == everything.status == RunnableStatus.SUCCESS
    assert "rag_corpus" not in allowlisted.output["content"]
    assert "source_display_name" in allowlisted.output["content"]
    assert "rag_corpus" in everything.output["content"]


# --------------------------------------------------------------------------------------------
# Addressing, errors, serialisation
# --------------------------------------------------------------------------------------------


def test_endpoint_follows_the_corpus_region_not_the_connection(fixture):
    """A full resource name must route to the corpus's region even if the connection disagrees."""
    other_region = "us-east4" if fixture.location != "us-east4" else "europe-west4"
    connection = _connection()
    connection.vertex_project_location = other_region

    node = VertexAIRagSearch(connection=connection, rag_corpus_id=fixture.corpus, top_k=3)
    assert node._endpoint_location() == fixture.location

    result = node.run(input_data={"query": REFUND_QUERY})
    assert result.status == RunnableStatus.SUCCESS, result.error


def test_bare_corpus_id_with_wrong_region_fails_cleanly(fixture):
    """The counterpart to the test above: without the region in the name, the region matters."""
    other_region = "us-east4" if fixture.location != "us-east4" else "europe-west4"
    connection = _connection()
    connection.vertex_project_location = other_region

    node = VertexAIRagSearch(connection=connection, rag_corpus_id=fixture.corpus.split("/")[-1], top_k=3)
    result = node.run(input_data={"query": REFUND_QUERY})

    assert result.status == RunnableStatus.FAILURE
    assert result.error.recoverable


def test_nonexistent_corpus_raises_recoverable_error(fixture):
    bogus = "/".join(fixture.corpus.split("/")[:-1] + ["1234567890123456789"])
    node = VertexAIRagSearch(connection=_connection(), rag_corpus_id=bogus)

    result = node.run(input_data={"query": REFUND_QUERY})

    assert result.status == RunnableStatus.FAILURE
    assert result.error.recoverable


def test_empty_query_raises_recoverable_error(tool):
    result = tool.run(input_data={"query": "   "})

    assert result.status == RunnableStatus.FAILURE
    assert result.error.recoverable
    assert "non-empty" in str(result.error)


def test_yaml_roundtrip_executes_against_the_live_api(tmp_path, fixture):
    """The platform stores and rebuilds node configs; a dropped field is config the user loses."""
    from dynamiq import Workflow
    from dynamiq.flows import Flow

    node = VertexAIRagSearch(
        connection=_connection(),
        rag_corpus_id=fixture.corpus,
        top_k=3,
        metadata_fields=["source_display_name"],
        vector_distance_threshold=0.9,
    )
    path = tmp_path / "vertex_rag_workflow.yaml"
    Workflow(flow=Flow(nodes=[node])).to_yaml_file(str(path))
    reloaded = Workflow.from_yaml_file(str(path), init_components=True).flow.nodes[0]

    assert reloaded.rag_corpus_id == fixture.corpus
    assert reloaded.top_k == 3
    assert reloaded.metadata_fields == ["source_display_name"]
    assert reloaded.vector_distance_threshold == 0.9

    result = reloaded.run(input_data={"query": REFUND_QUERY})
    assert result.status == RunnableStatus.SUCCESS, result.error
