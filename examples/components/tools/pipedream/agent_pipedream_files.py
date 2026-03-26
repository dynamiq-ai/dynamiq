"""
BEFORE RUNNING:
--------------
Make sure to specify the following fields in your YAML configuration file (pipedream_with_files.yaml):

1. authProvisionId: Your Pipedream authentication provision ID
2. stash_id: The ID of the stash to use for file storage (or "new", True, or empty string)
3. external_user_id: Your external user identifier for Pipedream Connect
"""

import os
from pathlib import Path

from dynamiq import Workflow
from dynamiq.callbacks import DynamiqTracingCallbackHandler
from dynamiq.connections.managers import get_connection_manager
from dynamiq.runnables import RunnableConfig


def save_files_from_result(result, output_dir: Path) -> None:
    """
    Extract and save files from workflow result.

    Args:
        result: Workflow execution result
        output_dir: Directory to save files to
    """
    # Navigate through nested output structure to get files
    files = result.output.get("output", {}).get("output", {}).get("output")

    if not files:
        print("\nNo files found in result.")
        return

    print(f"\nSaving {len(files)} file(s)...")

    for idx, file in enumerate(files):
        if hasattr(file, "name"):
            filename = file.name
        else:
            filename = f"file_{idx}"

        file_path = output_dir / filename

        with open(file_path, "wb") as f:
            file.seek(0)
            f.write(file.read())

        print(f"  ✓ Saved: {file_path}")

    print(f"\nAll files saved to: {output_dir.absolute()}")


def main():
    """Main execution function."""
    yaml_file_path = os.path.join(os.path.dirname(__file__), "pipedream_with_files.yaml")
    output_dir = Path(os.path.dirname(__file__)) / "files"
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Pipedream Agent - File Extraction Example")
    print("=" * 60)

    with get_connection_manager() as cm:
        wf = Workflow.from_yaml_file(file_path=yaml_file_path, connection_manager=cm, init_components=True)

        # Setup tracing callback
        trace = DynamiqTracingCallbackHandler(api_key=os.getenv("DYNAMIQ_ACCESS_KEY"))

        # Run workflow
        print("\nExecuting workflow...")
        result = wf.run(
            input_data={"input": "Extract attachment file from latest email"}, config=RunnableConfig(callbacks=[trace])
        )

        print("\nWorkflow Result:")
        print(result)

        save_files_from_result(result, output_dir)


if __name__ == "__main__":
    main()
