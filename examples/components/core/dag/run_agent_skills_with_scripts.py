"""Run the agent that uses FileSystem skills and runs mermaid-tools scripts in the E2B sandbox."""

import io
import os
from pathlib import Path

from dynamiq import Workflow
from dynamiq.connections.managers import get_connection_manager
from dynamiq.runnables import RunnableConfig, RunnableResult

EXAMPLES_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXAMPLES_DIR.parent.parent.parent.parent


def get_files_from_workflow_result(result: RunnableResult) -> list[io.BytesIO]:
    """Extract sandbox (or file-store) returned files from a workflow run result.

    The flow returns node_id -> to_dict(); agent output["files"] is preserved (BytesIO).
    Returns a flat list of all BytesIO files from any node that returned "files".
    """
    out_files: list[io.BytesIO] = []
    if not result.output:
        return out_files
    for _node_id, node_result in result.output.items():
        out = node_result.get("output") if isinstance(node_result, dict) else getattr(node_result, "output", None)
        if isinstance(out, dict) and out.get("files"):
            for f in out["files"]:
                if hasattr(f, "getvalue") and hasattr(f, "read"):
                    out_files.append(f)
    return out_files


SAMPLE_MARKDOWN = """# Sample doc

## Flow

```mermaid
flowchart LR
    A[User] --> B[Agent]
    B --> C[Diagram]
```
"""


def run_agent_skills_with_scripts():
    """Load workflow from YAML, upload sample.md + script, run agent to use mermaid-tools and run script in sandbox.

    Returns:
        Tuple of (workflow, run_result, list of BytesIO files from sandbox/output).
    """
    os.chdir(PROJECT_ROOT)
    result = None
    returned_files: list[io.BytesIO] = []

    yaml_path = EXAMPLES_DIR / "agent_skills_with_scripts.yaml"
    script_path = PROJECT_ROOT / ".skills" / "mermaid-tools" / "scripts" / "extract_diagrams.py"

    sample_bio = io.BytesIO(SAMPLE_MARKDOWN.encode("utf-8"))
    sample_bio.name = "input/sample.md"
    script_bio = io.BytesIO(script_path.read_text(encoding="utf-8").encode("utf-8"))
    script_bio.name = "input/mermaid-tools/scripts/extract_diagrams.py"
    uploaded_files = [sample_bio, script_bio]

    with get_connection_manager() as cm:
        wf = Workflow.from_yaml_file(
            file_path=yaml_path,
            connection_manager=cm,
            init_components=True,
        )

        prompt = (
            "Use the mermaid-tools skill and run its extraction script in the sandbox. "
            "1) List skills, then get the 'mermaid-tools' skill (note the scripts_path). "
            "2) In the sandbox, run exactly: python3 /home/user/input/mermaid-tools/scripts/"
            "extract_diagrams.py /home/user/input/sample.md /home/user/output. "
            "3) Report the full script output (stdout and stderr) and list any files "
            "created in /home/user/output. Output files there are automatically downloaded."
        )
        result = wf.run(
            input_data={
                "input": prompt,
                "files": uploaded_files,
            },
            config=RunnableConfig(callbacks=[]),
        )

        returned_files = get_files_from_workflow_result(result)
        if returned_files:
            print(f"\n=== Downloaded from sandbox ({len(returned_files)} file(s)) ===")
            for f in returned_files:
                name = getattr(f, "name", None) or getattr(f, "path", "unknown")
                size = len(f.getvalue()) if hasattr(f, "getvalue") else 0
                print(f"  - {name} ({size} bytes)")
    return wf, result, returned_files


if __name__ == "__main__":
    print("=== Skills with scripts (FileSystem: humanizer, mermaid-tools, markdown-tools, xlsx) ===\n")
    run_agent_skills_with_scripts()
    print("\nDone.")
