import io
from pathlib import Path

from dynamiq import Workflow
from dynamiq.connections import Neo4j
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.tools import CypherExecutor
from dynamiq.nodes.types import Behavior, InferenceMode
from dynamiq.utils.logger import logger
from examples.components.tools.custom_tools.file_reader import FileReaderTool
from examples.llm_setup import setup_llm

BRIEF_PATH = Path(__file__).parent / "data" / "graph_brief.md"

QUERY_ROLE = (
    "You answer questions using Neo4j. If schema is unknown, call cypher_executor with mode=introspect. "
    "Then use mode=execute with allow_writes=false. Use routing='r' for read queries when supported.\n"
    "- Use the XML protocol exactly with <action>cypher_executor</action> and <action_input> JSON.\n"
    "- Avoid comma-separated MATCH patterns; use explicit relationships.\n"
    "- If a local brief is available via the file reader, use it to clarify product names, aliases, or context.\n"
    "- When returning answers, include the Document ids that support each claim."
)


def _optional_brief_reader(path: Path) -> FileReaderTool | None:
    if not path.exists():
        logger.warning("Brief file %s does not exist; continuing without it.", path)
        return None

    content = path.read_bytes()
    buffer = io.BytesIO(content)
    buffer.description = path.name  # type: ignore[attr-defined]
    return FileReaderTool(files=[buffer])


def build_reader_agent() -> Agent:
    llm = setup_llm(model_provider="gpt", model_name="gpt-4o", temperature=0, max_tokens=2048)
    connection = Neo4j()
    cypher_executor = CypherExecutor(connection=connection, name="cypher_executor")

    tools = [cypher_executor]
    brief_reader = _optional_brief_reader(BRIEF_PATH)
    if brief_reader:
        tools.append(brief_reader)

    return Agent(
        name="graph_query_agent",
        description="Reads Neo4j and optional brief content to answer questions.",
        role=QUERY_ROLE,
        llm=llm,
        tools=tools,
        inference_mode=InferenceMode.FUNCTION_CALLING,
        max_loops=8,
        behaviour_on_max_loops=Behavior.RETURN,
    )


def run_queries() -> list[dict]:
    agent = build_reader_agent()
    workflow = Workflow(flow=Flow(nodes=[agent]))

    questions = [
        "Who uses the Dynamiq Platform, and for what use case (with their location)?",
        "GraphStream adoption leaderboard (topics + renewal term), sorted by most topics",
        "Dynamiq partner ecosystem: who they partner with and why (counted by mentions)",
        "Reference architecture” tech map: what tech is used for which purpose (RBAC, ingest, observability, etc.)",
    ]

    outputs = []
    for question in questions:
        logger.info("=== Query === %s", question)
        result = workflow.run(input_data={"input": question})
        agent_output = result.output.get(agent.id, {}).get("output", {})
        outputs.append({"question": question, "output": agent_output.get("content")})
    return outputs


if __name__ == "__main__":
    results = run_queries()
    for result in results:
        print(f"Q: {result['question']}\nA: {result['output']}\n")
