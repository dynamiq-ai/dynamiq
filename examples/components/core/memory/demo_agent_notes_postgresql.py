"""Agent notes backed by PostgreSQL.

A note is a title, a one-line description and a body. The agent always sees an index of
`title - description` for every note it owns, injected into its system prompt on every run,
and loads full bodies on demand with `read_note`.

This demo runs the agent twice with the same `user_id`. The second run is a fresh
conversation with no history — the agent knows the notes exist only because the index is in
its prompt.

Requires: OPENAI_API_KEY and a reachable PostgreSQL (POSTGRESQL_HOST / POSTGRESQL_PORT /
POSTGRESQL_DATABASE / POSTGRESQL_USER / POSTGRESQL_PASSWORD, defaulting to localhost:5432).
"""

from __future__ import annotations

import logging
import os

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.connections import PostgreSQL as PostgreSQLConnection
from dynamiq.memory.notes import NotesConfig
from dynamiq.memory.notes.backends.postgres import PostgresNotesBackend
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import OpenAI as OpenAILLM

LOGGER = logging.getLogger(__name__)

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
NOTES_TABLE_NAME = os.getenv("POSTGRESQL_NOTES_TABLE", "dynamiq_agent_notes")
USER_ID = os.getenv("DEMO_USER_ID", "demo-user-1")

FIRST_TURN = (
    "Save a note for later: our deploy process is merge to main, wait for CI green, "
    "./deploy staging, smoke-test, then ./deploy prod. Rollback is ./deploy rollback --last."
)
SECOND_TURN = "How do I roll back a bad deploy?"


def build_agent(backend: PostgresNotesBackend) -> Agent:
    llm = OpenAILLM(connection=OpenAIConnection(), model=OPENAI_MODEL)
    return Agent(
        name="notes-demo-agent",
        llm=llm,
        tools=[],
        notes=NotesConfig(backend=backend),
    )


def show(label: str, result) -> None:
    """Print an agent result, including the reason when it failed."""
    print(f"\n=== {label} ===")
    if result.output is None:
        print(f"FAILED: {result.error}")
        return
    print(result.output.get("content"))


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    backend = PostgresNotesBackend(
        connection=PostgreSQLConnection(),
        table_name=NOTES_TABLE_NAME,
    )

    # Fail fast with a clear message rather than letting every tool call error mid-run.
    backend.ensure_table()

    agent = build_agent(backend)

    # Run 1 — the agent decides to call write_note.
    show("RUN 1", agent.run(input_data={"input": FIRST_TURN, "user_id": USER_ID}))

    print("\n=== NOTES NOW STORED ===")
    for note in backend.list_all(user_id=USER_ID):
        print(f"- {note.title} - {note.description} ({len(note.content)} chars)")

    # Run 2 — a fresh conversation. No history is passed; the agent knows the note exists
    # only from the index in its system prompt, and must call read_note to answer.
    show("RUN 2 (fresh conversation)", agent.run(input_data={"input": SECOND_TURN, "user_id": USER_ID}))

    backend.close()


if __name__ == "__main__":
    main()
