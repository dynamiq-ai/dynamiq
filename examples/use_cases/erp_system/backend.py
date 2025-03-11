import os

from database_tool import DatabaseQueryTool, SQLiteConnection

from dynamiq.callbacks.streaming import StreamingIteratorCallbackHandler
from dynamiq.connections import HttpApiKey
from dynamiq.memory import Memory
from dynamiq.memory.backends.in_memory import InMemory
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.runnables import RunnableConfig
from dynamiq.types.streaming import StreamingConfig, StreamingMode
from examples.llm_setup import setup_llm
from examples.use_cases.smm_manager.mailgun_tool import MailGunTool

PERSONALITY = """
I am Dynamiq Assistant, your smart and friendly ERP System Assistant.
I’m efficient, detail-oriented, and here to help you manage your ERP system with ease.
I excel at simplifying complex database
interactions, providing clear and actionable insights, and ensuring data accuracy.
Let’s streamline your workflow together!
"""

ROLE = """
Some of the key responsibilities of the Dynamiq Assistant include:
- Validate user queries against the database schema before execution.
- Retrieve and manipulate ERP system data, including inventory, orders, and customer records.
- Automatically list available database tables and content when relevant.
- Ensure data integrity and prevent unauthorized updates or deletions.
- Provide user-friendly summaries of query results, such as inventory levels or sales statistics.
- Guide the user with suggestions for further actions, like refining their query or exploring related data.
"""

AGENT_ROLE = PERSONALITY + ROLE


def setup_agent() -> ReActAgent:
    """
    Initializes an AI agent with a specified role and streaming configuration.
    """
    db_connection = SQLiteConnection(db_path="mock_erp_system.db")

    # Create an instance of the DatabaseQueryTool
    db_tool = DatabaseQueryTool(
        connection=db_connection,
        name="ERP Database Query Tool",
        description="Query ERP database to retrieve and manipulate data.",
    )

    # Create connection
    connection_mailgun = HttpApiKey(api_key=os.getenv("MAILGUN_API_KEY"), url="https://api.mailgun.net/v3")

    # Create tool instance
    mailgun_tool = MailGunTool(connection=connection_mailgun, domain_name=os.getenv("MAILGUN_DOMAIN"))

    llm = setup_llm()
    memory = Memory(backend=InMemory())
    streaming_config = StreamingConfig(enabled=True, mode=StreamingMode.FINAL, by_tokens=True)

    agent = ReActAgent(
        name="Dynamiq Assistant",
        llm=llm,
        tools=[db_tool, mailgun_tool],
        memory=memory,
        streaming=streaming_config,
    )

    return agent


def generate_agent_response(agent: ReActAgent, user_input: str):
    """
    Processes the user input using the agent. Supports both streaming and non-streaming responses.
    """
    if agent.streaming.enabled:
        streaming_handler = StreamingIteratorCallbackHandler()
        agent.run(input_data={"input": user_input}, config=RunnableConfig(callbacks=[streaming_handler]))

        response_text = ""

        for chunk in streaming_handler:
            print(chunk)
            content = chunk.data.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if content:
                response_text += " " + content
                yield " " + content

    else:
        result = agent.run({"input": user_input})
        response_text = result.output.get("content", "")
        yield response_text
