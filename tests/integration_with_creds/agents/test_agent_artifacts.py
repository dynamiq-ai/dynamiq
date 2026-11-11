"""An agent publishes a deliverable and another reads it (OPENAI_API_KEY required).

``ArtifactAPISimulator`` stands in for the platform, so everything above the socket is real. The tests of
the backend and the artifact tool alone, which need no model, are in tests/integration/artifacts.
"""

import pytest

from dynamiq.artifacts import ArtifactConfig
from dynamiq.artifacts.backends import Dynamiq as DynamiqArtifacts
from dynamiq.connections import Dynamiq as DynamiqConnection
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.storages.file import FileStoreConfig, InMemoryFileStore
from tests.helpers.artifact_api import API_URL, ArtifactAPISimulator

MODEL = "gpt-5.4"

EMPLOYEE = "Dana Okafor"
MTTR = "37 minutes"
RATING = "Exceeds expectations"
REVIEW_REQUEST = f"""I'm finishing the Q3 performance reviews for the platform team and need the record \
for {EMPLOYEE} (Senior Platform Engineer) as a single HTML page I can share with HR and her.

Her Q3 numbers:
- Production deploys shipped: 142
- Incidents led as incident commander: 3, mean time to recovery {MTTR}
- Pull requests reviewed: 211
- Goal "Migrate the billing database to Postgres 16": completed two weeks early
- Goal "Mentor two new hires through onboarding": completed
- Manager rating: {RATING}

Include a short summary paragraph, a metrics table, the goals with their status, and the rating. \
Keep it clean and self-contained."""

READ_REQUEST = (
    f"HR is asking about {EMPLOYEE}'s Q3 performance record. What was her mean time to recovery on "
    "incidents, and what rating did she get? Answer from the record itself."
)


@pytest.fixture(scope="module")
def openai_llm():
    return OpenAI(model=MODEL, connection=OpenAIConnection())


@pytest.fixture(scope="module")
def run_config():
    return RunnableConfig()


@pytest.fixture
def api(monkeypatch):
    """The artifact API, served by the simulator for the duration of one test."""
    simulator = ArtifactAPISimulator()
    monkeypatch.setattr(DynamiqConnection, "connect", lambda self: simulator)
    return simulator


@pytest.fixture
def backend(api):
    return DynamiqArtifacts(connection=DynamiqConnection(url=API_URL, api_key="simulated-key"))


@pytest.mark.flaky(reruns=2)
@pytest.mark.integration
def test_a_deliverable_is_published_then_read_by_another_agent(openai_llm, run_config, backend, api):
    """One agent writes a performance record in its workspace and publishes it; a fresh one reads it.

    Nothing in the first request mentions artifacts: the agent has to recognise a shareable HTML
    page as one. The second agent gets no id and no shared conversation, only the same backend, so
    it has to find the record and read it rather than guess.
    """
    writer = Agent(
        name="ReviewWriter",
        llm=openai_llm,
        role="You are an engineering manager's assistant who prepares polished review documents.",
        inference_mode=InferenceMode.FUNCTION_CALLING,
        max_loops=12,
        file_store=FileStoreConfig(enabled=True, backend=InMemoryFileStore(), agent_file_write_enabled=True),
        artifacts=ArtifactConfig(enabled=True, backend=backend),
    )
    written = writer.run(input_data={"input": REVIEW_REQUEST}, config=run_config)

    assert written.status == RunnableStatus.SUCCESS, written.error
    assert len(api.artifacts) == 1, f"Expected one artifact, the server holds {len(api.artifacts)}."
    ((artifact_id, record),) = api.artifacts.items()
    html = api.latest_text(artifact_id)
    assert record["kind"] == "html", f"Published as {record['kind']}, not html."
    for fact in (EMPLOYEE, "142", MTTR.split()[0], "211", RATING):
        assert fact.lower() in html.lower(), f"'{fact}' is missing from the published record: {html[:500]}"
    assert record["versions"][-1]["client"].startswith("dynamiq-python/"), "The platform records the client."

    refs = written.output.get("artifacts") or []
    assert [(ref["id"], ref["version_id"]) for ref in refs] == [
        (artifact_id, record["versions"][-1]["id"])
    ], f"Run output did not report the deliverable's latest version: {refs}"
    assert html not in written.output["content"], "The answer should link the record, not paste it."

    reader = Agent(
        name="ReviewReader",
        llm=openai_llm,
        role="You answer HR questions from the team's published documents.",
        inference_mode=InferenceMode.FUNCTION_CALLING,
        max_loops=8,
        file_store=FileStoreConfig(enabled=True, backend=InMemoryFileStore(), agent_file_write_enabled=True),
        artifacts=ArtifactConfig(enabled=True, backend=backend),
    )
    calls_before = len(api.calls)
    answered = reader.run(input_data={"input": READ_REQUEST}, config=run_config)

    assert answered.status == RunnableStatus.SUCCESS, answered.error
    answer = answered.output["content"]
    reads = api.calls[calls_before:]
    content_reads = [p for verb, p in reads if verb == "GET" and p.startswith(f"/{artifact_id}/versions/")]
    assert content_reads, f"The record was never read. API calls: {reads}. Answer: {answer}"
    assert MTTR.split()[0] in answer, f"MTTR not taken from the record: {answer}"
    assert "exceed" in answer.lower(), f"Rating not taken from the record: {answer}"
    assert "artifacts" not in answered.output, "Reading an artifact must not report it as a deliverable."
    assert len(api.artifacts) == 1, "The reader must not publish anything."
