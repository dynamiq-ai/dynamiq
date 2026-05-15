import os

from dynamiq import Workflow
from dynamiq.connections.managers import get_connection_manager
from dynamiq.utils.logger import logger


def main() -> None:
    yaml_path = os.path.join(os.path.dirname(__file__), "ontology_memory_agent.yaml")
    logger.info("Loading ontology memory YAML workflow from %s", yaml_path)

    with get_connection_manager() as connection_manager:
        workflow = Workflow.from_yaml_file(
            file_path=yaml_path,
            connection_manager=connection_manager,
            init_components=True,
        )
        result = workflow.run(
            input_data={
                "input": "I prefer concise technical answers and I work at OpenAI.",
                "user_id": "demo-user",
                "session_id": "demo-session",
            }
        )

    print(result.output)


if __name__ == "__main__":
    main()
