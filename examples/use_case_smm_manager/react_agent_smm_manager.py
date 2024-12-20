import io
import os

from mailgun_tool import MailGunTool

from dynamiq.connections import E2B, HttpApiKey
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.tools.e2b_sandbox import E2BInterpreterTool
from examples.llm_setup import setup_llm


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

    file_io.name = filename if filename else "uploaded_file"
    file_io.description = description if description else "No description provided"

    return file_io


# Create connection
connection = HttpApiKey(api_key=os.getenv("MAILGUN_API_KEY"), url="https://api.mailgun.net/v3")

# Create tool instance
mailgun_tool = MailGunTool(connection=connection, domain_name=os.getenv("MAILGUN_DOMAIN"))
connection_e2b = E2B()

tool_code = E2BInterpreterTool(connection=connection_e2b)
FILE_PATH = ".data/emails.txt"


llm = setup_llm()


agent = ReActAgent(
    name="AI Agent",
    llm=llm,
    tools=[mailgun_tool, tool_code],
)
file_io = read_file_as_bytesio(
    file_path=FILE_PATH,
)

result = agent.run(
    input_data={
        "input": (
            "Read the file and craft a personalized email for each recipient. "
            "Also, send the emails to the recipients. "
            "End each email with the following text: 'Best regards, AI Agent from Dynamiq.'"
        ),
        "files": [file_io],
    },
    config=None,
)
print(result.output)
