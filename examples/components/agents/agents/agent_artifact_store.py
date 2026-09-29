"""Agent that publishes artifacts to the Dynamiq platform.

An artifact is a deliverable the user opens, reviews and shares: a named, typed document with
immutable versions and a link that outlives the conversation. The agent reaches it through one tool
with actions create, update, get and list. Binaries and office formats still go out as output files.

The first run publishes an HTML report. The second run gives the agent the artifact id and asks for
a change, which becomes v2 of the same artifact rather than a new one.

Requires ``DYNAMIQ_API_KEY`` (and ``DYNAMIQ_URL`` if not the default).
"""

import os

from dynamiq.connections import Dynamiq as DynamiqConnection
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.types import InferenceMode
from dynamiq.storages.artifact import ArtifactStoreConfig, DynamiqArtifactStore
from dynamiq.utils.logger import logger
from examples.llm_setup import setup_llm

PROJECT_ID = os.getenv("DYNAMIQ_PROJECT_ID")

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
        artifact_store=ArtifactStoreConfig(
            enabled=True,
            # Unset project_id: artifacts belong to the user behind the API key.
            backend=DynamiqArtifactStore(connection=DynamiqConnection(), project_id=PROJECT_ID),
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
