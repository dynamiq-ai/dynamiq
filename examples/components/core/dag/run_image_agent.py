import io
import logging
import os
from pathlib import Path

from dynamiq import Workflow, runnables
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections.managers import get_connection_manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_workflow_from_yaml() -> Workflow:
    """Load the image agent workflow from YAML file."""
    yaml_file_path = os.path.join(os.path.dirname(__file__), "image_agent_workflow.yaml")

    if not os.path.exists(yaml_file_path):
        raise FileNotFoundError(
            f"YAML workflow file not found: {yaml_file_path}\n"
            f"Please ensure 'image_agent_workflow.yaml' is in the same directory."
        )

    logger.info(f"Loading workflow from: {yaml_file_path}")

    with get_connection_manager() as cm:
        wf = Workflow.from_yaml_file(file_path=yaml_file_path, connection_manager=cm, init_components=True)

    logger.info(f"Workflow loaded: {wf.id}")

    return wf


def load_image_file(file_path: str) -> io.BytesIO:
    """Load an image file into a BytesIO object."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {file_path}")

    with open(path, "rb") as f:
        image_bytes = f.read()

    image_file = io.BytesIO(image_bytes)
    image_file.name = path.name
    return image_file


def save_image_files(files: list[io.BytesIO], output_dir: str, prefix: str = "output") -> list[str]:
    """Save image files to disk."""
    if not files:
        return []

    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []

    logger.info(f"Saved {len(files)} image(s):")
    for i, file in enumerate(files):
        if hasattr(file, "name") and file.name:
            file_name = f"{prefix}_{i}_{Path(file.name).name}"
        else:
            file_name = f"{prefix}_{i}.png"

        output_path = os.path.join(output_dir, file_name)

        file.seek(0)
        with open(output_path, "wb") as f:
            f.write(file.read())

        saved_paths.append(output_path)
        logger.info(f"  - {output_path}")

    return saved_paths


def main():
    output_dir = "./generated_images"

    workflow = load_workflow_from_yaml()
    agent = workflow.flow.nodes[0]

    logger.info("Image AI Assistant - Type 'quit' to exit\n")

    while True:
        try:
            file_path_input = input("File path (press Enter to skip): ").strip()

            if file_path_input.lower() in ("quit", "exit", "q"):
                break

            files = []
            if file_path_input:
                for path in file_path_input.split(","):
                    path = path.strip()
                    if path:
                        try:
                            file = load_image_file(path)
                            files.append(file)
                        except Exception as e:
                            logger.error(f"Error loading {path}: {e}")

            task_input = input("Task: ").strip()

            if task_input.lower() in ("quit", "exit", "q"):
                break

            if not task_input:
                continue

            input_data = {"input": task_input}
            if files:
                for f in files:
                    f.seek(0)
                input_data["files"] = files

            tracer = TracingCallbackHandler()
            result = workflow.run(
                input_data=input_data,
                config=runnables.RunnableConfig(callbacks=[tracer]),
            )

            if result.status == runnables.RunnableStatus.SUCCESS:
                agent_output = result.output.get(agent.id, {}).get("output", {})
                content = agent_output.get("content", "")
                output_files = agent_output.get("files", [])

                logger.info(f"{content}\n")

                if output_files:
                    save_image_files(output_files, output_dir)
            else:
                logger.error(f"Error: {result.error}\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error: {e}\n")


if __name__ == "__main__":
    main()
