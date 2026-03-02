import argparse
import os
from io import BytesIO
from pathlib import Path

from dynamiq import Workflow
from dynamiq.callbacks import DynamiqTracingCallbackHandler
from dynamiq.connections.managers import get_connection_manager
from dynamiq.runnables import RunnableConfig

EXAMPLES_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXAMPLES_DIR.parent.parent.parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "scripts" / "output"


PROMPT_DOCX = (
    "Create a multi-page Word document report on a topic of your choice "
    "(e.g. market analysis, project summary, or research findings). "
    "Include a title, executive summary, at least one table and one chart, "
    "section headings and a few short paragraphs or bullets per section."
)

PROMPT_PPTX = (
    "Create a 15-slide PowerPoint presentation on a topic of your choice "
    "(e.g. climate change, digital transformation, or healthcare innovation). "
    "Include title slide, agenda, 3–4 sections with key points and examples, "
    "summary slide and closing with next steps or call to action."
)

PROMPT_PDF = (
    "Create a multi-page PDF report on a topic of your choice "
    "(e.g. market overview, project brief, or research summary). "
    "Include a title, short intro, section headings and a few paragraphs or bullets per section."
)

PROMPT_XLSX = (
    "Create an Excel workbook with simulated data sheets (timestamps, outcomes, categories), "
    "a dashboard sheet with key metrics and formulas, at least one chart, "
    "and clean tabular data suitable for export to CSV or R."
)

PROMPT_FULL_REPORT = (
    "Choose a single topic (e.g. climate strategy, digital transformation, healthcare analytics) "
    "and create a full report package: Excel workbook with data and dashboard, "
    "10–15 slide presentation, long-form Word document with executive summary and sections, "
    "and a comprehensive PDF that ties the narrative and visuals together."
)

PROMPT_FRONTEND_DESIGN = (
    "Create a distinctive landing page or small web app (e.g. product, event, or portfolio) "
    "with a bold, non-generic aesthetic. Focus on typography, color, and layout. "
    "Avoid generic AI aesthetics."
)

PROMPT_WEB_ARTIFACTS_BUILDER = (
    "Create a multi-component web app with React, TypeScript, Vite, and shadcn/ui "
    "(e.g. dashboard, configurator, or multi-step form). "
    "Produce a single bundle for viewing; optionally provide a live preview URL."
)

PROMPT_RECREATE_GETDYNAMIQ_WEBSITE = (
    "Recreate a simplified version of the Dynamiq marketing website (getdynamiq.ai). "
    "Include hero with tagline 'Build agentic applications in a matter of hours', "
    "value propositions (ROI, dev time, compliance), feature grid, security section, and footer. "
    "Match professional, enterprise tone. Optionally provide a live preview URL."
)


DEFAULT_PROMPT = PROMPT_DOCX


def collect_files_from_workflow_result(result) -> list[tuple[str, BytesIO]]:
    """Collect all 'files' from workflow run result (e.g. sandbox output files).

    Returns:
        List of (filename, BytesIO) pairs. Filename is from BytesIO.name or a fallback.
    """
    collected: list[tuple[str, BytesIO]] = []
    output = getattr(result, "output", None) or {}
    if not isinstance(output, dict):
        return collected
    for node_id, node_result in output.items():
        if not isinstance(node_result, dict):
            continue
        node_output = node_result.get("output")
        if not isinstance(node_output, dict):
            continue
        files = node_output.get("files")
        if not files or not isinstance(files, list):
            continue
        for i, f in enumerate(files):
            if not isinstance(f, BytesIO):
                continue
            name = getattr(f, "name", None)
            if not name or not isinstance(name, str):
                name = f"file_{node_id}_{i}"
            else:
                name = Path(name).name
            collected.append((name, f))
    return collected


def save_workflow_files_locally(result, output_dir: Path | None = None) -> list[Path]:
    """Collect files from workflow result and save them under output_dir. Returns paths saved."""
    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for name, blob in collect_files_from_workflow_result(result):
        path = output_dir / name
        if path.exists():
            base, ext = path.stem, path.suffix
            n = 1
            while path.exists():
                path = output_dir / f"{base}_{n}{ext}"
                n += 1
        blob.seek(0)
        path.write_bytes(blob.read())
        saved.append(path)
    return saved


def run_agent_registry_skills_sandbox(
    prompt: str = DEFAULT_PROMPT,
    callbacks: list | None = None,
):
    """Load workflow from YAML and run agent with Dynamiq registry + E2B sandbox.

    Skills are ingested into the sandbox at init. Pass callbacks for tracing (e.g. UI).
    Sandbox output files (e.g. from /home/user/output) are collected and returned:
    result.output["<agent_node_id>"]["output"]["files"] is a list of BytesIO objects.
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


PROMPTS = {
    "docx": PROMPT_DOCX,
    "pdf": PROMPT_PDF,
    "pptx": PROMPT_PPTX,
    "xlsx": PROMPT_XLSX,
    "full_report": PROMPT_FULL_REPORT,
    "frontend_design": PROMPT_FRONTEND_DESIGN,
    "web_artifacts_builder": PROMPT_WEB_ARTIFACTS_BUILDER,
    "recreate_getdynamiq_website": PROMPT_RECREATE_GETDYNAMIQ_WEBSITE,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run registry skills + E2B sandbox workflow.")
    parser.add_argument(
        "--task",
        choices=list(PROMPTS),
        default="docx",
        help="Task: docx, pdf, pptx, xlsx, full_report; "
        "frontend_design, web_artifacts_builder, recreate_getdynamiq_website; "
        "task_1..task_4 (standalone prompts). Default: docx.",
    )
    args = parser.parse_args()
    prompt = PROMPTS[args.task]

    print("=== Registry skills + E2B sandbox (Dynamiq registry, auto-ingest) ===\n")
    print(f"Task: {args.task}\n")
    if os.environ.get("DYNAMIQ_TRACE_ACCESS_KEY"):
        wf, result, _ = run_with_ui_tracing(prompt=prompt)
    else:
        wf, result = run_agent_registry_skills_sandbox(prompt=prompt)
    saved = save_workflow_files_locally(result)
    if saved:
        print(f"\nSaved {len(saved)} file(s) from workflow:")
        for p in saved:
            print(f"  {p}")
    else:
        print("\nNo files collected from workflow.")
    print("\nDone.")
