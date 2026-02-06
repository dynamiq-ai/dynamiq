from pathlib import Path

from dynamiq import Workflow
from dynamiq.connections.managers import get_connection_manager
from dynamiq.runnables import RunnableConfig

EXAMPLES_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXAMPLES_DIR.parent.parent.parent.parent


def run_agent_local_skill_yaml():
    """Load workflow from YAML, run humanizer skill agent (FileSystem registry)."""
    import os

    os.chdir(PROJECT_ROOT)

    yaml_path = EXAMPLES_DIR / "agent_local_skill.yaml"

    with get_connection_manager() as cm:
        wf = Workflow.from_yaml_file(
            file_path=yaml_path,
            connection_manager=cm,
            init_components=True,
        )

        prompt = (
            "Show skills, then humanize text: "
            "In conclusion, it is important to note that leveraging cutting-edge solutions "
            "can help stakeholders unlock value and drive transformative outcomes. "
            "Moving forward, we will utilize best practices to ensure synergy."
        )
        wf.run(
            input_data={"input": prompt},
            config=RunnableConfig(callbacks=[]),
        )

    return wf


if __name__ == "__main__":
    print("=== Humanizer skill (YAML DAG, FileSystem registry) ===\n")
    run_agent_local_skill_yaml()
    print("\nDone.")
