import os

from dynamiq import Workflow, runnables
from dynamiq.connections.managers import get_connection_manager


def main():
    yaml_path = os.path.join(os.path.dirname(__file__), "agents_as_tools.yaml")
    with get_connection_manager() as cm:
        wf = Workflow.from_yaml_file(file_path=yaml_path, connection_manager=cm, init_components=True)

        result = wf.run(
            input_data={
                "input": "Compute the sum of the first 7 integers and explain briefly.",
                "tool_params": {"by_name": {"Coder Agent": {"global": {"n": 7}}}},
            },
            config=runnables.RunnableConfig(),
        )
        content = None
        if isinstance(result.output, dict):
            content = result.output.get("content")
            if content is None:
                try:
                    first = next(iter(result.output.values()))
                    if isinstance(first, dict):
                        content = first.get("output", {}).get("content")
                except StopIteration:
                    content = None
        print("Workflow result content:\n", content)


if __name__ == "__main__":
    main()
