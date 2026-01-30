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
    """Upload .skills/ into FileStore. Normalize SKILL.MD -> SKILL.md for discovery."""
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


def run_agent_xlsx_price_dashboard_yaml():
    """Load workflow from YAML, upload skills and data, run xlsx price dashboard."""
    logger.info("Loading xlsx price dashboard workflow from YAML...")
    yaml_path = os.path.join(EXAMPLES_DIR, "agent_xlsx_price_dashboard.yaml")

    with get_connection_manager() as cm:
        wf = Workflow.from_yaml_file(
            file_path=yaml_path,
            connection_manager=cm,
            init_components=True,
        )
        flow = wf.flow
        agent_node = flow._node_by_id.get("xlsx-agent")
        if agent_node and getattr(agent_node, "file_store", None) and agent_node.file_store.enabled:
            backend = agent_node.file_store.backend
            upload_skills_to_backend(backend)
            for tool in getattr(agent_node, "tools", []) or []:
                if hasattr(tool, "file_store") and getattr(tool, "file_store", None) is not None:
                    tool.file_store = backend
        else:
            logger.warning("Agent node or file_store not found")

        prompt = (
            "Create a house price simulation and dashboard in Excel based on this sample data. "
            "X (e.g. size sqft or bedrooms): 10, 15, 20, 25, 30, 35, 40, 45, 50. "
            "Y (price): 202000, 250000, 280000, 310000, 350000, 380000, 420000, 460000, 500000. "
            "First use the SkillsTool to get the "
            "xlsx skill content (README) so you know how to create Excel with openpyxl and formulas. "
            "Then use the code executor to"
            " build an Excel workbook with formulas (no hardcoded results):"
            "include the data, summary analytics, ROI calculation, "
            "and coefficient (e.g. slope/trend) so the dashboard stays dynamic."
            " Save the workbook so it is returned (e.g. generated/house_price_dashboard.xlsx)."
        )
        wf.run(
            input_data={"input": prompt},
            config=RunnableConfig(callbacks=[]),
        )

    logger.info("Workflow %s finished.", wf.id)
    return wf


if __name__ == "__main__":
    print("=== Xlsx price dashboard (YAML DAG) ===\n")
    run_agent_xlsx_price_dashboard_yaml()
    print("\nDone.")
