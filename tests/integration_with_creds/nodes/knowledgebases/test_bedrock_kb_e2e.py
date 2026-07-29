import os
import time
import uuid
from typing import NamedTuple

import pytest

from dynamiq.connections import AWS
from dynamiq.nodes.knowledgebases.bedrock import BedrockKnowledgeBaseSearch, BedrockManagedSearchConfig
from dynamiq.runnables import RunnableStatus

pytestmark = pytest.mark.integration

SAMPLE_DOCUMENT = (
    "Dynamiq refund policy: customers can request a full refund within 30 days of purchase. "
    "Refunds are processed by the support department within 5 business days. "
    "After 30 days, purchases are only eligible for account credit."
)
QUERY = "How many days do customers have to request a full refund?"

# Identities used by bootstrap mode, which ingests the sample document with a per-document ACL.
ALLOWED_USER = "dynamiq-e2e-allowed@example.com"
DENIED_USER = "dynamiq-e2e-denied@example.com"

CREATE_TIMEOUT_SECONDS = 600
INDEX_TIMEOUT_SECONDS = 600
POLL_INTERVAL_SECONDS = 10


class KnowledgeBaseUnderTest(NamedTuple):
    """The knowledge base being exercised, plus what the tests may assume about its contents."""

    id: str
    is_managed: bool
    allowed_user: str | None  # identity permitted to read the corpus, when ACLs are in play
    denied_user: str | None  # identity that must not; only known for a bootstrapped corpus


def _aws_available() -> bool:
    """Whether boto3 can resolve credentials and a region by any supported means.

    Deliberately not a check for AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY: those are unset when
    authenticating through an SSO or named profile (AWS_DEFAULT_PROFILE), which the AWS connection
    supports, and requiring them made this whole module skip silently for profile-based callers.
    """
    try:
        session = AWS().get_boto3_session()
        return session.get_credentials() is not None and session.region_name is not None
    except Exception:
        return False


def _wait_for(describe, is_ready, is_failed, timeout, step_name):
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = describe()
        if is_ready(state):
            return state
        if is_failed(state):
            raise AssertionError(f"{step_name} failed with state: {state}")
        time.sleep(POLL_INTERVAL_SECONDS)
    raise AssertionError(f"{step_name} did not complete within {timeout}s")


@pytest.fixture(scope="module")
def knowledge_base():
    """Provide the KB under test: an existing one from the env, or a bootstrapped managed KB.

    Bootstrap mode creates a fully managed knowledge base with an ACL-enabled custom connector data
    source, ingests a small inline document scoped to ALLOWED_USER, and tears everything down
    afterwards. It requires BEDROCK_KB_ROLE_ARN (an IAM role Bedrock can assume for the knowledge
    base). For an existing KB, set BEDROCK_KNOWLEDGE_BASE_ID and, if it is a MANAGED-type KB,
    BEDROCK_KNOWLEDGE_BASE_MANAGED=true; if that KB enforces ACLs, also set
    BEDROCK_KNOWLEDGE_BASE_USER_ID to an identity allowed to read it.
    """
    if not _aws_available():
        pytest.skip("AWS credentials/region could not be resolved")

    existing_kb_id = os.getenv("BEDROCK_KNOWLEDGE_BASE_ID")
    if existing_kb_id:
        is_managed = os.getenv("BEDROCK_KNOWLEDGE_BASE_MANAGED", "").lower() in ("1", "true", "yes")
        yield KnowledgeBaseUnderTest(
            id=existing_kb_id,
            is_managed=is_managed,
            allowed_user=os.getenv("BEDROCK_KNOWLEDGE_BASE_USER_ID"),
            denied_user=None,
        )
        return

    role_arn = os.getenv("BEDROCK_KB_ROLE_ARN")
    if not role_arn:
        pytest.skip("Set BEDROCK_KNOWLEDGE_BASE_ID (existing KB) or BEDROCK_KB_ROLE_ARN (bootstrap mode)")

    session = AWS().get_boto3_session()
    agent_client = session.client("bedrock-agent")
    suffix = uuid.uuid4().hex[:8]

    kb_response = agent_client.create_knowledge_base(
        name=f"dynamiq-e2e-{suffix}",
        description="Temporary KB created by dynamiq integration tests; safe to delete.",
        roleArn=role_arn,
        knowledgeBaseConfiguration={
            "type": "MANAGED",
            "managedKnowledgeBaseConfiguration": {"embeddingModelType": "MANAGED"},
        },
    )
    kb_id = kb_response["knowledgeBase"]["knowledgeBaseId"]
    data_source_id = None

    try:
        _wait_for(
            describe=lambda: agent_client.get_knowledge_base(knowledgeBaseId=kb_id)["knowledgeBase"]["status"],
            is_ready=lambda status: status == "ACTIVE",
            is_failed=lambda status: status in ("FAILED", "DELETE_UNSUCCESSFUL"),
            timeout=CREATE_TIMEOUT_SECONDS,
            step_name="Knowledge base creation",
        )

        # A MANAGED knowledge base rejects a bare "CUSTOM" data source type; the custom connector is
        # selected through connectorParameters on MANAGED_KNOWLEDGE_BASE_CONNECTOR instead.
        data_source_response = agent_client.create_data_source(
            knowledgeBaseId=kb_id,
            name=f"dynamiq-e2e-docs-{suffix}",
            dataSourceConfiguration={
                "type": "MANAGED_KNOWLEDGE_BASE_CONNECTOR",
                "managedKnowledgeBaseConnectorConfiguration": {
                    "connectorParameters": {"type": "CUSTOM", "version": "1", "aclEnabled": True}
                },
            },
        )
        data_source_id = data_source_response["dataSource"]["dataSourceId"]

        # CreateDataSource is asynchronous for managed KBs: CREATING -> AVAILABLE.
        _wait_for(
            describe=lambda: agent_client.get_data_source(knowledgeBaseId=kb_id, dataSourceId=data_source_id)[
                "dataSource"
            ]["status"],
            is_ready=lambda status: status in ("AVAILABLE", "ACTIVE"),
            is_failed=lambda status: status in ("FAILED", "DELETE_UNSUCCESSFUL"),
            timeout=CREATE_TIMEOUT_SECONDS,
            step_name="Data source creation",
        )

        document_id = f"dynamiq-e2e-doc-{suffix}"
        agent_client.ingest_knowledge_base_documents(
            knowledgeBaseId=kb_id,
            dataSourceId=data_source_id,
            documents=[
                {
                    "content": {
                        "dataSourceType": "CUSTOM",
                        "custom": {
                            "customDocumentIdentifier": {"id": document_id},
                            "sourceType": "IN_LINE",
                            "inlineContent": {"type": "TEXT", "textContent": {"data": SAMPLE_DOCUMENT}},
                        },
                    },
                    "metadata": {
                        "type": "IN_LINE_ATTRIBUTE",
                        "inlineAttributes": [
                            {"key": "department", "value": {"stringValue": "support", "type": "STRING"}}
                        ],
                        "accessControlList": [
                            {"name": ALLOWED_USER, "type": "USER", "access": "ALLOW"},
                            {"name": DENIED_USER, "type": "USER", "access": "DENY"},
                        ],
                    },
                }
            ],
        )

        def describe_document_status():
            response = agent_client.get_knowledge_base_documents(
                knowledgeBaseId=kb_id,
                dataSourceId=data_source_id,
                documentIdentifiers=[
                    {"dataSourceType": "CUSTOM", "custom": {"id": document_id}},
                ],
            )
            details = response.get("documentDetails") or []
            return details[0]["status"] if details else "NOT_FOUND"

        _wait_for(
            describe=describe_document_status,
            is_ready=lambda status: status == "INDEXED",
            is_failed=lambda status: status in ("FAILED", "IGNORED", "METADATA_UPDATE_FAILED"),
            timeout=INDEX_TIMEOUT_SECONDS,
            step_name="Document indexing",
        )

        yield KnowledgeBaseUnderTest(id=kb_id, is_managed=True, allowed_user=ALLOWED_USER, denied_user=DENIED_USER)
    finally:
        if data_source_id:
            try:
                agent_client.delete_data_source(knowledgeBaseId=kb_id, dataSourceId=data_source_id)
            except Exception as e:  # noqa: BLE001 - best-effort teardown
                print(f"Failed to delete data source {data_source_id}: {e}")
        try:
            agent_client.delete_knowledge_base(knowledgeBaseId=kb_id)
        except Exception as e:  # noqa: BLE001 - best-effort teardown
            print(f"Failed to delete knowledge base {kb_id}: {e}")


@pytest.fixture
def tool(knowledge_base):
    kwargs = {}
    if knowledge_base.is_managed:
        kwargs["managed_search"] = BedrockManagedSearchConfig()
    if knowledge_base.allowed_user:
        # ACL-enforced retrieval fails closed: without an identity the KB returns nothing at all.
        kwargs["user_id"] = knowledge_base.allowed_user
    return BedrockKnowledgeBaseSearch(connection=AWS(), knowledge_base_id=knowledge_base.id, top_k=5, **kwargs)


def test_retrieve_returns_relevant_documents(tool):
    result = tool.run(input_data={"query": QUERY})

    assert result.status == RunnableStatus.SUCCESS
    documents = result.output["documents"]
    assert documents, "expected at least one retrieved document"
    assert "30 days" in result.output["content"]
    top_document = documents[0]
    assert top_document.content
    assert top_document.metadata is not None


def test_retrieve_respects_top_k(tool):
    result = tool.run(input_data={"query": QUERY, "top_k": 1})

    assert result.status == RunnableStatus.SUCCESS
    assert len(result.output["documents"]) <= 1


def test_retrieve_enforces_document_acl(tool, knowledge_base):
    """A denied identity must retrieve nothing, and the node-level default must be overridable."""
    if not knowledge_base.denied_user:
        pytest.skip("ACL assertions need a bootstrapped corpus with known allowed/denied identities")

    allowed = tool.run(input_data={"query": QUERY, "user_id": knowledge_base.allowed_user})
    assert allowed.status == RunnableStatus.SUCCESS
    assert allowed.output["documents"], "the allowed identity should retrieve the document"

    denied = tool.run(input_data={"query": QUERY, "user_id": knowledge_base.denied_user})
    assert denied.status == RunnableStatus.SUCCESS
    assert denied.output["documents"] == [], "the denied identity must not retrieve the document"
    assert denied.output["content"] == "No results found for the given query."
