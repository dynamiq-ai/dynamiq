from pathlib import Path

from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.tools import PythonCodeExecutor
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.file import FileStoreConfig, InMemoryFileStore
from dynamiq.utils.logger import logger

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SKILLS_DIR = PROJECT_ROOT / ".skills"
SKILL_OUTPUTS_DIR = Path(__file__).resolve().parent / "skill_outputs"

SKILL_OUTPUTS_EXCLUDE = (
    ".skills/",
    "SKILL.md",
    "recalc.py",
    "build_price_analytics.py",
    "run.py",
    "house_prices.csv",
)
SKILL_OUTPUTS_REPORT_EXTENSIONS = (".xlsx", ".xls", ".pdf", ".docx", ".pptx")
SKILL_OUTPUTS_REPORT_PREFIXES = ("generated/", "report/", "output/")


def _is_report_output_file(name: str) -> bool:
    """True if this file should be saved to skill_outputs (report/output only)."""
    if not name:
        return False
    name_lower = name.lower()
    for exc in SKILL_OUTPUTS_EXCLUDE:
        if exc in name_lower or name_lower.endswith(exc):
            return False
    if any(name_lower.startswith(p) for p in SKILL_OUTPUTS_REPORT_PREFIXES):
        return True
    if any(name_lower.endswith(ext) for ext in SKILL_OUTPUTS_REPORT_EXTENSIONS):
        return True
    return False


AGENT_ROLE = """
You have access to skills and a Python code executor.
Use the SkillsTool to list skills and get the xlsx skill content
(README/SKILL) so you know how to create Excel workbooks with openpyxl
and formulas. Then use the code executor to write and run Python
that builds the Excel following the skill's guidelines (formulas,
not hardcoded values). Produce clear reports and summarize findings in
text. Format responses in Markdown.
"""


def upload_skills_to_filestore(file_store: InMemoryFileStore) -> None:
    """Upload .skills/ into FileStore. Normalize SKILL.MD -> SKILL.md so the loader discovers skills."""
    logger.info("Uploading skills to FileStore...")
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
                file_store.store(store_path, path.read_bytes())
                logger.info(" Uploaded %s", store_path)


def create_agent(file_store: InMemoryFileStore, tracing_handler=None) -> Agent:
    """Create agent with skills (SkillsTool) and PythonCodeExecutor so it uses skill README and code to build Excel."""
    file_store_config = FileStoreConfig(
        enabled=True,
        backend=file_store,
        agent_file_write_enabled=True,
    )
    python_tool = PythonCodeExecutor(name="python_executor", file_store=file_store)
    agent = Agent(
        name="XlsxDashboardAgent",
        llm=OpenAI(model="gpt-5-mini", temperature=0.3, max_tokens=4096),
        tools=[python_tool],
        role=AGENT_ROLE,
        max_loops=12,
        inference_mode=InferenceMode.XML,
        file_store=file_store_config,
        skills_enabled=True,
    )
    logger.info("Agent created with skills_enabled=True")
    return agent


def main():
    print("\n" + "=" * 80)
    print("Xlsx skill: house price simulation and dashboard (Excel formulas)")
    print("=" * 80 + "\n")

    file_store = InMemoryFileStore()
    upload_skills_to_filestore(file_store)

    tracing_handler = TracingCallbackHandler()
    agent = create_agent(file_store, tracing_handler)

    prompt = (
        "Create a house price simulation and dashboard in Excel based on this sample data. "
        "X (e.g. size sqft or bedrooms): 10, 15, 20, 25, 30, 35, 40, 45, 50. "
        "Y (price): 202000, 250000, 280000, 310000, 350000, 380000, 420000, 460000, 500000. "
        "First use the SkillsTool to get the"
        "xlsx skill content (README) so you know how to create Excel with openpyxl and formulas. "
        "Then use the code executor to build an Excel"
        "workbook with formulas (no hardcoded results): include the data, summary analytics, ROI calculation, "
        "and coefficient (e.g. slope/trend) so the dashboard stays"
        "dynamic. Save the workbook so it is returned"
        "(e.g. generated/house_price_dashboard.xlsx)."
    )
    print("Prompt:", prompt, "\n")

    input_data = {"input": prompt}
    callbacks = [tracing_handler] if tracing_handler else None
    run_config = RunnableConfig(callbacks=callbacks) if callbacks else None
    result = agent.run(input_data=input_data, config=run_config)
    output = result.output.get("content", "")
    files = result.output.get("files", [])

    print("=" * 80)
    print("Agent output")
    print("=" * 80)
    print(output)
    print("=" * 80)

    if files:
        file_list = files if isinstance(files, list) else list(files.values()) if isinstance(files, dict) else [files]
        report_files = [f for f in file_list if _is_report_output_file(getattr(f, "name", ""))]
        if report_files:
            SKILL_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
            for f in report_files:
                name = getattr(f, "name", "")
                base = name.split("/")[-1] if "/" in name else name or "output"
                path = SKILL_OUTPUTS_DIR / base
                if hasattr(f, "read"):
                    if hasattr(f, "seek"):
                        f.seek(0)
                    path.write_bytes(f.read())
                    print(f"Saved to skill_outputs: {path.name}")
        if len(report_files) < len(file_list):
            print(f"(Skipped {len(file_list) - len(report_files)} non-report file(s): internal skill/input files)")
    print()


if __name__ == "__main__":
    main()
