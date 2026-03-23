import os

from dynamiq.connections import E2B, HttpApiKey
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.tools.e2b_sandbox import E2BInterpreterTool

from dynamiq.nodes.tools.mailgun import MailGunTool
from examples.utils import read_file_as_bytesio, setup_llm

# Create connection
connection = HttpApiKey(api_key=os.getenv("MAILGUN_API_KEY"), url="https://api.mailgun.net/v3")

# Create tool instance
mailgun_tool = MailGunTool(connection=connection, domain_name=os.getenv("MAILGUN_DOMAIN"))
connection_e2b = E2B()

tool_code = E2BInterpreterTool(connection=connection_e2b)
FILE_PATH = "emails.txt"


llm = setup_llm()


agent = Agent(
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
            "Read the file and craft the personalised email for each recipient. "
            "Also send the emails to the recipients."
            "For end of the email use the following text: 'Best regards, AI Agent from Dynamiq'"
        ),
        "files": [file_io],
    },
    config=None,
)
print(result.output)
