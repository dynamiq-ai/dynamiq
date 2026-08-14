"""Persistent notes surviving context compaction.

The agent is prompted (system block + role) to save critical values (keys,
tokens, IDs) to a dedicated notes file in its sandbox. When conversation
history is compacted, the compaction observation tells it to re-read that
file, so exact values survive even though the raw history was replaced by
an LLM summary.

The example gives the agent a deployment token, forces compaction mid-run with
a low token ceiling, then checks the token is still present in the final answer.

Requires E2B_API_KEY (persistent notes are sandbox-only) and an LLM API key.

What to look for in the output:
  1. "Persistent Notes block in system prompt: yes"
  2. INFO log "Token limit exceeded. Automatically invoking Context Manager Tool."
  3. agent_notes.md exists in the sandbox and contains the token
  4. PASS: final answer still contains the token after compaction
"""

import logging

from dynamiq.connections import E2B
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.agents.utils import SummarizationConfig
from dynamiq.sandboxes import SandboxConfig
from dynamiq.sandboxes.e2b import E2BSandbox
from examples.llm_setup import setup_llm

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

SECRET = "XK-9137-QUARTZ"  # nosec B105 - demo placeholder, not a real credential

AGENT_ROLE = """Diligent assistant that completes multi-step file work.

Critical rule: whenever the user provides an exact value that will be needed later
(API key, token, credential, ID, URL), your FIRST action - before any other work -
is to append it verbatim to your persistent notes file. Conversation history gets
compacted and can drop exact values; sandbox files persist, and the notes file is
the dedicated place to keep them."""


def main():
    sandbox = E2BSandbox(connection=E2B())

    agent = Agent(
        name="persistent-notes-agent",
        llm=setup_llm(temperature=0),
        role=AGENT_ROLE,
        sandbox=SandboxConfig(enabled=True, backend=sandbox),
        summarization_config=SummarizationConfig(
            enabled=True,
            max_token_context_length=6000,  # low ceiling so compaction fires mid-run
            max_preserved_tokens=500,  # keep almost no raw history verbatim
        ),
        max_loops=25,
    )

    ops_block = agent.system_prompt_manager._prompt_blocks.get("operational_instructions", "")
    print(f"\nPersistent Notes block in system prompt: {'yes' if 'Persistent Notes' in ops_block else 'NO'}")
    notes_path = agent.get_notes_file_path()
    print(f"Advertised notes file: {notes_path}\n")

    try:
        result = agent.run(
            input_data={
                "input": (
                    f"My deployment token is {SECRET}. You will need it at the very end.\n\n"
                    "Now do this work step by step, one file per step:\n"
                    "write 6 files named story_1.md ... story_6.md, each containing a different "
                    "~400-word short story about space exploration.\n\n"
                    "After ALL six files are written, tell me the deployment token I gave you "
                    "at the start, formatted exactly as: TOKEN=<value>"
                )
            }
        )

        compacted = any("history was compacted" in (getattr(m, "content", "") or "") for m in agent._prompt.messages)
        print(f"\n--- Compaction happened during run: {'yes' if compacted else 'NO (lower max_token_context_length)'}")

        if sandbox.exists(notes_path):
            notes = sandbox.retrieve(notes_path).decode("utf-8")
            print(f"--- {notes_path} content:\n{notes}")
            print(f"--- Secret saved in notes file: {'yes' if SECRET in notes else 'no'}")
        else:
            print(f"--- {notes_path} was NOT created (agent skipped note-taking)")

        answer = str(result.output.get("content", ""))
        print(f"\n--- Final answer:\n{answer[:500]}\n")
        print("PASS: secret survived compaction" if SECRET in answer else "FAIL: secret lost")
    finally:
        try:
            sandbox.close(kill=True)
            print("\nE2B sandbox closed.")
        except Exception as e:
            print(f"\nFailed to close E2B sandbox: {e}")


if __name__ == "__main__":
    main()
