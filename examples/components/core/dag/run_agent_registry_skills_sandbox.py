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
    "Create a concise, multi-page Word document report on a topic of your choice "
    "(e.g. market analysis, project summary, or research findings). "
    "Include: a clear title and optional subtitle; "
    "an executive summary or intro; at least one table "
    "(e.g. key metrics, comparison, or data summary) with headers and a few rows; "
    "at least one chart or graphic; section headings and 2–4 "
    "short paragraphs or bullet lists per section so the report spans multiple pages. "
    "Keep wording concise."
)

PROMPT_PPTX = (
    "Create a 15-slide PowerPoint presentation on a topic of your choice "
    "(e.g. climate change, digital transformation, or healthcare innovation). "
    "Structure: title slide; agenda; "
    "3–4 sections with 2–3 slides each (intro, key points, evidence or examples); "
    "slide transitions, clear headings, bullet points or short takeaways; "
    "summary slide and a closing slide with next steps or call to action."
)

PROMPT_PDF = (
    "Create a concise, multi-page PDF report on a topic of your choice "
    "(e.g. market overview, project brief, or research summary). "
    "Include: a clear title; a short summary or intro; "
    "section headings and 2–4 short paragraphs or "
    "bullet lists per section so the report spans several pages."
)

PROMPT_XLSX = (
    "Create an Excel workbook that includes: "
    "(1) Data simulation: one or more sheets with "
    "simulated experiment data (e.g. random or formula-based: timestamps, numeric outcomes, "
    "categories, or time series) with clear column headers and enough rows to be meaningful. "
    "(2) Dashboard: a summary sheet with key metrics "
    "(totals, averages, counts) and references to the data sheets. "
    "(3) At least one chart (e.g. line, bar, or pie) "
    "based on the data, embedded in the workbook. "
    "(4) R-ready export: raw data sheet(s) in clean tabular layout "
    "(no merged cells in the data area, headers in first row) "
    "so the data can be exported to CSV and used in R; "
    "optionally add a sheet named 'README' or 'R_usage' with a short description of columns and how to load in R."
)

PROMPT_FULL_REPORT = (
    "Choose a single topic (e.g. climate strategy, digital "
    "transformation, healthcare analytics, or supply chain resilience) "
    "and create a full report package on that topic.\n\n"
    "1. Excel workbook: simulated data sheets, a dashboard sheet "
    "(key metrics, formulas), and at least one chart.\n\n"
    "2. Presentation: about 10–15 slides summarizing the topic—title, "
    "agenda, sections with key points and takeaways, summary and next steps.\n\n"
    "3. Word narrative: long-form document with executive summary, "
    "introduction, several sections (methodology, data overview, findings, "
    "dashboard summary, charts discussion, recommendations, conclusion). "
    "Include tables and references to the dashboard and charts. "
    "Aim for content that fills roughly 25–30 pages when exported to PDF.\n\n"
    "4. PDF report: a single, comprehensive PDF (approximately 30 pages) "
    "that combines the narrative and ties in the dashboard and presentation."
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
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run registry skills + E2B sandbox workflow.")
    parser.add_argument(
        "--task",
        choices=list(PROMPTS),
        default="docx",
        help="Task: docx, pdf, pptx, xlsx, or full_report "
        "(30-page PDF + Excel + presentation + Word on one topic). Default: docx.",
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
