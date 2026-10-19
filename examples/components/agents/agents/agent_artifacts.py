"""Agent that publishes artifacts to the Dynamiq platform.

An artifact is a deliverable the user opens, reviews and shares: a named, typed document with
immutable versions and a link that outlives the conversation. Artifacts move as files: the agent
writes a file in its workspace and publishes it with 'create', and changes one by loading it with
'get', editing the saved copy and publishing it with 'update' and the artifact's id. That workspace is
the agent's sandbox or file store, here an in-memory file store the agent can write to.

The first run publishes an HTML report. The second run gives the agent the artifact id and asks for
a change: it loads the report, edits it and publishes it as v2 of the same artifact.

Requires ``DYNAMIQ_API_KEY`` (and ``DYNAMIQ_URL`` if not the default). With a personal access token
the artifacts go to the artifact store named by ``DYNAMIQ_ARTIFACT_STORE_ID``; a conversation token
can leave it unset, and the artifacts then belong to its user.
"""

import os

from dynamiq.artifacts import ArtifactConfig
from dynamiq.artifacts.backends import Dynamiq as DynamiqArtifacts
from dynamiq.connections import Dynamiq as DynamiqConnection
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.types import InferenceMode
from dynamiq.storages.file import FileStoreConfig, InMemoryFileStore
from dynamiq.utils.logger import logger
from examples.llm_setup import setup_llm

ARTIFACT_STORE_ID = os.getenv("DYNAMIQ_ARTIFACT_STORE_ID")

FIRST_RUN = (
    "Build a one-page HTML report on the quarterly pipeline: Q1 $1.2M, Q2 $1.5M, Q3 $1.9M. "
    "Include a simple inline SVG bar chart."
)
SECOND_RUN = "In the report {artifact_id}, add a Q4 forecast of $2.3M and mark it as a forecast."


def agent_with_artifacts() -> Agent:
    return Agent(
        name="AgentWithArtifacts",
        llm=setup_llm(),
        role="You are an analyst who delivers polished, self-contained reports.",
        inference_mode=InferenceMode.FUNCTION_CALLING,
        max_loops=8,
        file_store=FileStoreConfig(enabled=True, backend=InMemoryFileStore(), agent_file_write_enabled=True),
        artifacts=ArtifactConfig(
            enabled=True,
            backend=DynamiqArtifacts(connection=DynamiqConnection(), artifact_store_id=ARTIFACT_STORE_ID),
        ),
    )


def run(agent: Agent, query: str) -> list[dict]:
    result = agent.run(input_data={"input": query})
    output = result.output or {}
    logger.info(f"Answer: {output.get('content') or result.error}")
    for ref in output.get("artifacts", []):
        logger.info(f"Artifact {ref['id']} v{ref['version']} ({ref['kind']}): {ref['url']}")
    return output.get("artifacts", [])


if __name__ == "__main__":
    print("--- First run: publish ---")
    artifacts = run(agent_with_artifacts(), FIRST_RUN)

    if artifacts:
        print("\n--- Second run: revise the same artifact ---")
        run(agent_with_artifacts(), SECOND_RUN.format(artifact_id=artifacts[0]["id"]))
