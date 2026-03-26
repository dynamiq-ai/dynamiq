import os

from dynamiq import Workflow
from dynamiq.connections.managers import get_connection_manager

if __name__ == "__main__":
    yaml_file_path = os.path.join(os.path.dirname(__file__), "config.yaml")

    with get_connection_manager() as cm:
        wf = Workflow.from_yaml_file(file_path=yaml_file_path, connection_manager=cm, init_components=True)

        result = wf.run(
            input_data={
                "input": "Use the Cua Desktop tool to open YouTube, \
                    search for 'Eurovision', \
                    and extract the title of the first 10 videos shown in the results."
            }
        )
        print(result)
