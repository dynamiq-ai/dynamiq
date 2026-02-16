import os
from pathlib import Path

from dynamiq import Workflow
from dynamiq.callbacks import DynamiqTracingCallbackHandler, TracingCallbackHandler
from dynamiq.connections.managers import get_connection_manager
from dynamiq.runnables import RunnableConfig

EXAMPLES_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXAMPLES_DIR.parent.parent.parent.parent

DEFAULT_INPUT_YAML = EXAMPLES_DIR / "agent_registry_skills_sandbox.yaml"
DEFAULT_OUTPUT_YAML = EXAMPLES_DIR / "agent_registry_skills_sandbox_dump.yaml"
DEFAULT_PROMPT = (
    "List available skills, then get the mermaid-tools skill. "
    "Run its extraction script in the sandbox (scripts_path will be under /home/user/skills/mermaid-tools/scripts). "
    "Use a small markdown snippet with a mermaid diagram as input and report the script output."
)


def run_workflow_yaml_cycle(
    input_yaml: str | Path = DEFAULT_INPUT_YAML,
    output_yaml: str | Path = DEFAULT_OUTPUT_YAML,
    prompt: str = DEFAULT_PROMPT,
    callbacks: list | None = None,
):
    """Load, dump, reload, and run both workflows to validate YAML cycle."""
    os.chdir(PROJECT_ROOT)
    input_yaml = Path(input_yaml)
    output_yaml = Path(output_yaml)
    callbacks = callbacks or []

    with get_connection_manager() as cm:
        workflow_original = Workflow.from_yaml_file(
            file_path=input_yaml,
            connection_manager=cm,
            init_components=True,
        )
        workflow_original.to_yaml_file(output_yaml)

        workflow_loaded = Workflow.from_yaml_file(
            file_path=output_yaml,
            connection_manager=cm,
            init_components=True,
        )

        result_original = workflow_original.run(
            input_data={"input": prompt},
            config=RunnableConfig(callbacks=callbacks),
        )
        result_loaded = workflow_loaded.run(
            input_data={"input": prompt},
            config=RunnableConfig(callbacks=callbacks),
        )

    return workflow_original, result_original, workflow_loaded, result_loaded


def run_workflow_yaml_cycle_with_ui_tracing(
    input_yaml: str | Path = DEFAULT_INPUT_YAML,
    output_yaml: str | Path = DEFAULT_OUTPUT_YAML,
    prompt: str = DEFAULT_PROMPT,
    base_url: str | None = None,
    access_key: str | None = None,
):
    """Run YAML cycle with Dynamiq UI tracing callback."""
    base_url = base_url or os.environ.get("DYNAMIQ_TRACE_BASE_URL", "https://collector.sandbox.getdynamiq.ai")
    access_key = access_key or os.environ.get("DYNAMIQ_TRACE_ACCESS_KEY")
    tracing = DynamiqTracingCallbackHandler(base_url=base_url, access_key=access_key)
    results = run_workflow_yaml_cycle(
        input_yaml=input_yaml,
        output_yaml=output_yaml,
        prompt=prompt,
        callbacks=[tracing],
    )
    return (*results, tracing)


if __name__ == "__main__":
    print("=== Registry skills + sandbox YAML cycle ===")
    print(f"Input YAML: {DEFAULT_INPUT_YAML}")
    print(f"Dump YAML: {DEFAULT_OUTPUT_YAML}")

    if os.environ.get("DYNAMIQ_TRACE_ACCESS_KEY"):
        run_workflow_yaml_cycle_with_ui_tracing()
    else:
        run_workflow_yaml_cycle(callbacks=[TracingCallbackHandler()])

    print("Done.")
