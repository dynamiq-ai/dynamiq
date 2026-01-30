import os
from pathlib import Path

from dynamiq import Workflow
from dynamiq.connections.managers import get_connection_manager
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

EXAMPLES_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXAMPLES_DIR.parent.parent.parent.parent
SKILLS_DIR = PROJECT_ROOT / ".skills"


def upload_skills_to_backend(backend):
    """Upload .skills/ into the agent's FileStore backend."""
    if not SKILLS_DIR.exists():
        logger.warning("No .skills directory at %s", SKILLS_DIR)
        return
    for skill_dir in SKILLS_DIR.iterdir():
        if not skill_dir.is_dir():
            continue
        for path in skill_dir.rglob("*"):
            if path.is_file():
                rel = path.relative_to(SKILLS_DIR)
                store_path = f".skills/{rel.as_posix()}"
                if store_path.endswith("/SKILL.MD"):
                    store_path = store_path.replace("/SKILL.MD", "/SKILL.md")
                backend.store(store_path, path.read_bytes())
                logger.info(" Uploaded %s", store_path)


def run_agent_skills_yaml():
    """Load workflow from YAML, upload skills to FileStore, run once."""
    logger.info("Loading agent-with-skills workflow from YAML...")
    yaml_path = os.path.join(EXAMPLES_DIR, "agent_skills.yaml")

    with get_connection_manager() as cm:
        wf = Workflow.from_yaml_file(
            file_path=yaml_path,
            connection_manager=cm,
            init_components=True,
        )
        flow = wf.flow
        agent_node = flow._node_by_id.get("skills-agent")
        if agent_node and getattr(agent_node, "file_store", None) and agent_node.file_store.enabled:
            upload_skills_to_backend(agent_node.file_store.backend)
        else:
            logger.warning("Agent node or file_store not found; skills may be empty")

        prompt = (
            "1) List available skills. "
            "2) Get the full content of the hello-world skill. "
            "3) If the skill has a script,"
            " run it with run_script "
            "(skill_name=hello-world, script_path=scripts/run.py). "
            "Summarize what the skill is for and what the script output was."
        )
        wf.run(
            input_data={"input": prompt},
            config=RunnableConfig(callbacks=[]),
        )

    logger.info("Workflow %s finished.", wf.id)
    for node_id, result in wf.flow._results.items():
        logger.info("Node %s: %s", node_id, str(result)[:200])
    return wf


if __name__ == "__main__":
    print("=== Agent with skills (YAML DAG) ===\n")
    wf = run_agent_skills_yaml()
    print("\nDone.")
