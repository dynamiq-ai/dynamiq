import asyncio
import logging
import os
from pathlib import Path

from dynamiq import Workflow, runnables
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections.managers import get_connection_manager

logger = logging.getLogger(__name__)

YAML_PATH = Path(__file__).with_name("neo4j_text2cypher_agent.yaml")


def _run(workflow_id: str, input_data: dict, callbacks: list | None = None):
    with get_connection_manager() as cm:
        wf = Workflow.from_yaml_file(
            file_path=str(YAML_PATH),
            wf_id=workflow_id,
            connection_manager=cm,
            init_components=True,
        )
        result = wf.run(input_data=input_data, config=runnables.RunnableConfig(callbacks=callbacks or []))
    logger.info("Workflow %s finished. Output: %s", workflow_id, result.output)
    return result


def run_reader():
    return _run(
        workflow_id="neo4j-reader-workflow",
        input_data={"input": "Who works at Dynamiq? Return names and roles if present."},
        callbacks=[TracingCallbackHandler()],
    )


def run_ingest():
    return _run(
        workflow_id="neo4j-ingest-workflow",
        input_data={
            "input": "Add that Charlie works at Dynamiq as a Product Manager, then list everyone who works at Dynamiq."
        },
        callbacks=[TracingCallbackHandler()],
    )


async def run_ingest_async():
    with get_connection_manager() as cm:
        wf = Workflow.from_yaml_file(
            file_path=str(YAML_PATH),
            wf_id="neo4j-ingest-workflow",
            connection_manager=cm,
            init_components=True,
        )
        result = await wf.run(
            input_data={
                "input": "Add that Charlie works at Dynamiq "
                "as a Product Manager, then list everyone who works at Dynamiq."
            },
            config=runnables.RunnableConfig(callbacks=[TracingCallbackHandler()]),
        )
    logger.info("Async ingest workflow finished. Output: %s", result.output)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    if not os.environ.get("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY is not set. Set it before running.")

    run_reader()
    run_ingest()
    asyncio.run(run_ingest_async())
