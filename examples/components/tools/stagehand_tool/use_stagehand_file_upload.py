import io
import os

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.connections.managers import get_connection_manager
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.nodes.types import InferenceMode
from dynamiq.storages.file import FileStoreConfig
from dynamiq.storages.file.in_memory import InMemoryFileStore

from dynamiq.connections import Stagehand as StagehandConnection
from dynamiq.nodes.tools import Stagehand


def set_wf_with_agent(cm):
    stagehand_tool = Stagehand(
        connection=StagehandConnection(
            model_api_key=os.getenv("OPENAI_API_KEY"),
        ),
        model_name="gpt-4o",
        is_postponed_component_init=True,
    )

    llm = OpenAI(
        id="openai",
        connection=OpenAIConnection(),
        model="gpt-4o",
        temperature=0.3,
        max_tokens=1000,
        is_postponed_component_init=True,
    )

    file_store_config = FileStoreConfig(enabled=True, backend=InMemoryFileStore())

    agent = Agent(
        id="agent",
        llm=llm,
        tools=[stagehand_tool],
        role="assistant",
        inference_mode=InferenceMode.XML,
        max_loops=10,
        file_store=file_store_config,
    )

    return agent


def read_file_as_bytesio(file_path: str, filename: str = None, description: str = None) -> io.BytesIO:
    """
    Reads the content of a file and returns it as a BytesIO object with custom attributes for filename and description.

    Args:
        file_path (str): The path to the file.
        filename (str, optional): Custom filename for the BytesIO object.
        description (str, optional): Custom description for the BytesIO object.

    Returns:
        io.BytesIO: The file content in a BytesIO object with custom attributes.
    """
    with open(file_path, "rb") as f:
        file_content = f.read()

    file_io = io.BytesIO(file_content)

    file_io.name = filename if filename else "uploaded_file.csv"
    file_io.description = description if description else "No description provided"

    return file_io


FILE_PATH = ""


def main():
    with get_connection_manager() as cm:
        agent = set_wf_with_agent(cm)

        png_file_to_upload = read_file_as_bytesio(
            file_path=FILE_PATH,
            filename="test.csv",
            description="CSV to convert",
        )

        agent.run(
            input_data={
                "input": (
                    "Use the Stagehand tool to open https://cloudconvert.com/csv-to-xls and convert provided file."
                ),
                "files": [png_file_to_upload],
            },
        )
        print("Agent finished")


if __name__ == "__main__":
    main()
