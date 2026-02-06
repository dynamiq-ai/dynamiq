"""Run agent with Dynamiq API skill (humanizer) from YAML.

Loads agent_humanizer_skill.yaml, runs with a humanizer prompt. Set DYNAMIQ_URL
(e.g. https://api.sandbox.getdynamiq.ai) and DYNAMIQ_API_KEY.
"""

from pathlib import Path

from dynamiq import Workflow
from dynamiq.connections.managers import get_connection_manager
from dynamiq.runnables import RunnableConfig

EXAMPLES_DIR = Path(__file__).resolve().parent


def run_agent_humanizer_skill_yaml():
    """Load workflow from YAML, run humanizer skill agent."""
    yaml_path = EXAMPLES_DIR / "agent_humanizer_skill.yaml"

    with get_connection_manager() as cm:
        wf = Workflow.from_yaml_file(
            file_path=yaml_path,
            connection_manager=cm,
            init_components=True,
        )

        prompt = (
            "Show your skills "
            "Humanize this text: "
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
    print("=== Humanizer skill (YAML DAG) ===\n")
    run_agent_humanizer_skill_yaml()
    print("\nDone.")
