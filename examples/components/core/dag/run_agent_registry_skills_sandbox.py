"""Run the agent that uses skills from the Dynamiq registry and E2B sandbox.

Skills are ingested at workflow init: each configured skill (by skill id and version id)
is downloaded as zip, unzipped, and uploaded to the sandbox at /home/user/skills/<name>/.
The agent uses SkillsTool to get instructions and scripts_path (e.g. /home/user/skills/mermaid-tools/scripts).

To be ready to run:
  1. Export DYNAMIQ_URL (e.g. https://api.sandbox.getdynamiq.ai), DYNAMIQ_API_KEY, DYNAMIQ_PROJECT_ID.
  2. Run: python scripts/local_skill_upload_download_test.py --local-skill-dir .skills/mermaid-tools
     and use the printed skill id and version id in agent_registry_skills_sandbox.yaml (skills.source.skills).
  3. Export E2B_API_KEY and ANTHROPIC_API_KEY.
  4. Run this module or the workflow with init_components=True.
"""

import os
from pathlib import Path

from dynamiq import Workflow
from dynamiq.callbacks import DynamiqTracingCallbackHandler
from dynamiq.connections.managers import get_connection_manager
from dynamiq.runnables import RunnableConfig

EXAMPLES_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXAMPLES_DIR.parent.parent.parent.parent

DEFAULT_PROMPT = (
    "List available skills, then get the mermaid-tools skill. "
    "Run its extraction script in the sandbox (scripts_path will be under /home/user/skills/mermaid-tools/scripts). "
    "Use a small markdown snippet with a mermaid diagram as input and report the script output."
)


def run_agent_registry_skills_sandbox(
    prompt: str = DEFAULT_PROMPT,
    callbacks: list | None = None,
):
    """Load workflow from YAML and run agent with Dynamiq registry + E2B sandbox.

    Skills are ingested into the sandbox at init. Pass callbacks for tracing (e.g. UI).
    """
    os.chdir(PROJECT_ROOT)

    yaml_path = EXAMPLES_DIR / "agent_registry_skills_sandbox.yaml"
    with get_connection_manager() as cm:
        wf = Workflow.from_yaml_file(
            file_path=yaml_path,
            connection_manager=cm,
            init_components=True,
        )
        result = wf.run(
            input_data={"input": prompt},
            config=RunnableConfig(callbacks=callbacks or []),
        )
    return wf, result


def run_with_ui_tracing(
    prompt: str = DEFAULT_PROMPT,
    base_url: str | None = None,
    access_key: str | None = None,
):
    """Set DYNAMIQ_TRACE_ACCESS_KEY (and optional DYNAMIQ_TRACE_BASE_URL)."""
    base_url = base_url or os.environ.get("DYNAMIQ_TRACE_BASE_URL", "https://collector.sandbox.getdynamiq.ai")
    access_key = access_key or os.environ.get("DYNAMIQ_TRACE_ACCESS_KEY")
    tracing = DynamiqTracingCallbackHandler(base_url=base_url, access_key=access_key)
    wf, result = run_agent_registry_skills_sandbox(prompt=prompt, callbacks=[tracing])
    return wf, result, tracing


if __name__ == "__main__":
    print("=== Registry skills + E2B sandbox (Dynamiq registry, auto-ingest) ===\n")
    if os.environ.get("DYNAMIQ_TRACE_ACCESS_KEY"):
        run_with_ui_tracing()
    else:
        run_agent_registry_skills_sandbox()
    print("\nDone.")
