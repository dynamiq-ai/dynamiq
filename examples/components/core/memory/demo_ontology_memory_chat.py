import argparse
import logging
import os
import warnings
from pathlib import Path

from dynamiq.memory import Memory, MemorySaveMode
from dynamiq.memory.backends.in_memory import InMemory
from dynamiq.memory.backends.sqlite import SQLite
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.tools.ontology_memory import OntologyMemoryTool, OntologyMemoryToolInputSchema
from dynamiq.nodes.types import InferenceMode
from dynamiq.storages.graph import InMemoryGraphStore
from examples.llm_setup import setup_llm

warnings.filterwarnings(
    "ignore",
    message=r"Using extra keyword arguments on `Field` is deprecated and will be removed\..*",
    category=DeprecationWarning,
)


USER_ID = "demo-user"
SESSION_ID = "demo-session"
USER_LABEL = "Alex"
SHOW_FRAMEWORK_LOGS = os.getenv("ONTOLOGY_DEMO_VERBOSE_LOGS", "").lower() in {"1", "true", "yes"}
DEFAULT_AGENT_MEMORY_DB = "examples/components/core/memory/.demo_agent_chat_memory.sqlite"
DEFAULT_ONTOLOGY_STATE_FILE = "examples/components/core/memory/.demo_ontology_state.json"

AGENT_ROLE = """You are a helpful assistant.

When ontology memory context is provided, use it when relevant.
Do not invent user facts that are not present in the provided context or normal conversation.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive ontology memory agent demo.")
    parser.add_argument(
        "--ontology",
        choices=["on", "off"],
        default="on",
        help="Enable ontology-backed memory ingestion and context injection.",
    )
    parser.add_argument(
        "--agent-memory",
        choices=["none", "in-memory", "sqlite"],
        default="none",
        help="Enable the agent's normal chat memory backend.",
    )
    parser.add_argument(
        "--agent-memory-sqlite-path",
        default=DEFAULT_AGENT_MEMORY_DB,
        help="SQLite path for agent chat memory when --agent-memory=sqlite.",
    )
    parser.add_argument(
        "--ontology-store-file",
        default=None,
        help="Persist ontology graph state locally as JSON. Example: " f"{DEFAULT_ONTOLOGY_STATE_FILE}",
    )
    parser.add_argument(
        "--reset-ontology-store",
        action="store_true",
        help="Clear the persisted ontology state file before starting.",
    )
    parser.add_argument(
        "--reset-agent-memory",
        action="store_true",
        help="Delete the agent SQLite memory file before starting.",
    )
    return parser.parse_args()


def configure_demo_logging() -> None:
    if SHOW_FRAMEWORK_LOGS:
        return
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("dynamiq").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)


def print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def resolve_agent_memory(args: argparse.Namespace) -> Memory | None:
    if args.agent_memory == "none":
        return None
    if args.agent_memory == "in-memory":
        return Memory(backend=InMemory(), save_mode=MemorySaveMode.INPUT_OUTPUT)
    if args.reset_agent_memory:
        Path(args.agent_memory_sqlite_path).unlink(missing_ok=True)
    return Memory(
        backend=SQLite(db_path=args.agent_memory_sqlite_path, index_name="ontology_demo_chat"),
        save_mode=MemorySaveMode.INPUT_OUTPUT,
    )


def resolve_graph_store(args: argparse.Namespace) -> InMemoryGraphStore | None:
    if args.ontology == "off":
        return None
    state_file = args.ontology_store_file
    if state_file and args.reset_ontology_store:
        Path(state_file).unlink(missing_ok=True)
    return InMemoryGraphStore(state_file=state_file)


def setup_chat_agent(args: argparse.Namespace) -> tuple[Agent, OntologyMemoryTool | None]:
    agent_llm = setup_llm(temperature=0.2, max_tokens=1200)
    graph_store = resolve_graph_store(args)

    ontology_tool = None
    tools = []
    if graph_store is not None:
        extractor_llm = setup_llm(temperature=0, max_tokens=1200)
        ontology_tool = OntologyMemoryTool(
            client=object(),
            graph_store=graph_store,
            llm=extractor_llm,
            ensure_schema_on_init=False,
            memory_query_limit=50,
            name="ontology_memory",
        )
        ontology_tool.init_components()
        tools.append(ontology_tool)

    agent = Agent(
        name="Ontology Memory Agent",
        id="ontology-memory-agent",
        llm=agent_llm,
        tools=tools,
        memory=resolve_agent_memory(args),
        role=AGENT_ROLE,
        inference_mode=InferenceMode.XML,
        max_loops=6,
        verbose=False,
    )
    agent.init_components()
    return agent, ontology_tool


def print_demo_configuration(args: argparse.Namespace) -> None:
    print("Ontology Memory Agent Chat Demo")
    print(f"- Ontology memory: {args.ontology}")
    print(f"- Agent chat memory: {args.agent_memory}")
    print(f"- Ontology local persistence: {args.ontology_store_file or 'off'}")
    if args.agent_memory == "sqlite":
        print(f"- Agent SQLite memory path: {args.agent_memory_sqlite_path}")
    print("Type 'exit' to stop.\n")


def print_commit_summary(commit: dict) -> None:
    committed = commit.get("commit", {})
    print_section("MEMORY INGEST")
    print(f"Episode: {commit['episode']['id']}")
    print(f"Entities extracted: {len(committed.get('entities', []))}")
    print(f"Facts extracted: {len(committed.get('facts', []))}")
    notes = committed.get("notes", [])
    if notes:
        print("Notes:")
        for note in notes:
            print(f"- {note}")


def print_agent_answer(answer: str | None) -> None:
    print_section("AGENT ANSWER")
    print(answer or "")


def store_user_turn(ontology_tool: OntologyMemoryTool, user_input: str) -> dict:
    return ontology_tool.execute(
        OntologyMemoryToolInputSchema(
            operation="add_episode",
            content=user_input,
            user_id=USER_ID,
            session_id=SESSION_ID,
            metadata={"user_label": USER_LABEL},
            auto_extract=True,
        )
    )


def build_context_block(ontology_tool: OntologyMemoryTool, user_input: str) -> str:
    result = ontology_tool.execute(
        OntologyMemoryToolInputSchema(
            operation="get_context_block",
            query=user_input,
            user_id=USER_ID,
            session_id=SESSION_ID,
        )
    )
    return result["context_block"]


def print_agent_memory_state(agent: Agent) -> None:
    if agent.memory is None:
        return
    print_section("AGENT CHAT MEMORY")
    messages = agent.memory.get_all(limit=8)
    if not messages:
        print("- No agent chat memory stored.")
        return
    for message in messages:
        print(f"- {message.role.value}: {message.content}")


def print_ontology_memory_state(ontology_tool: OntologyMemoryTool) -> None:
    facts_result = ontology_tool.execute(
        OntologyMemoryToolInputSchema(
            operation="search_facts",
            user_id=USER_ID,
            session_id=SESSION_ID,
            include_inactive=True,
            limit=20,
        )
    )
    context_result = ontology_tool.execute(
        OntologyMemoryToolInputSchema(
            operation="get_context_block",
            query="What should the assistant remember about this user?",
            user_id=USER_ID,
            session_id=SESSION_ID,
            limit=20,
        )
    )

    print_section("CURRENT FACTS")
    if not facts_result["facts"]:
        print("- No facts stored.")
    for fact in facts_result["facts"]:
        print(
            f"- {fact['subject_label']} --{fact['predicate']}--> "
            f"{fact.get('object_label') or fact.get('object_value')} [{fact['status']}]"
        )

    print_section("CONTEXT BLOCK")
    print(context_result["context_block"])


def build_agent_input(user_input: str, ontology_tool: OntologyMemoryTool | None) -> str:
    if ontology_tool is None:
        return user_input
    context_block = build_context_block(ontology_tool, user_input)
    return (
        "ONTOLOGY MEMORY CONTEXT\n"
        f"{context_block}\n\n"
        "USER MESSAGE\n"
        f"{user_input}\n\n"
        "Answer the user naturally. Use ontology memory when it is relevant."
    )


def print_turn_state(agent: Agent, ontology_tool: OntologyMemoryTool | None) -> None:
    print_agent_memory_state(agent)
    if ontology_tool is not None:
        print_ontology_memory_state(ontology_tool)


def chat_loop(args: argparse.Namespace) -> None:
    configure_demo_logging()
    agent, ontology_tool = setup_chat_agent(args)
    print_demo_configuration(args)

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            break
        if not user_input:
            continue

        if ontology_tool is not None:
            commit = store_user_turn(ontology_tool, user_input)
            print_commit_summary(commit)

        response = agent.run(
            {
                "input": build_agent_input(user_input, ontology_tool),
                "user_id": USER_ID,
                "session_id": SESSION_ID,
            }
        )
        print_agent_answer(response.output.get("content"))
        print_turn_state(agent, ontology_tool)


if __name__ == "__main__":
    chat_loop(parse_args())
