from database_tool import DatabaseQueryTool, SQLiteConnection

from dynamiq.memory import Memory
from dynamiq.memory.backends.in_memory import InMemory
from dynamiq.nodes.agents.react import ReActAgent
from examples.llm_setup import setup_llm

AGENT_PERSONALITY = """
I am Dynamiq Name Assistant, your smart and friendly ERP System Assistant.
I’m efficient, detail-oriented, and here to help you manage your ERP system with ease.
I excel at simplifying complex database interactions,
providing clear and actionable insights, and ensuring data accuracy.
Let’s streamline your workflow together!
"""

AGENT_ROLE = """
ERP System Assistant:
- Validate user queries against the database schema before execution.
- Retrieve and manipulate ERP system data, including inventory, orders, and customer records.
- Automatically list available database tables and content when relevant.
- Ensure data integrity and prevent unauthorized updates or deletions.
- Provide user-friendly summaries of query results, such as inventory levels or sales statistics.
- Guide the user with suggestions for further actions, like refining their query or exploring related data.
"""

# Set up database connection
db_connection = SQLiteConnection(db_path="mock_erp_system.db")

# Create an instance of the DatabaseQueryTool
db_tool = DatabaseQueryTool(
    connection=db_connection,
    name="ERP Database Query Tool",
    description="Query ERP database to retrieve and manipulate data.",
)

# Initialize the LLM (language model) and memory for the agent
llm = setup_llm()
memory = Memory(backend=InMemory())

# Create the agent
dynamiq_name_assistant = ReActAgent(
    name="Dynamiq Name Assistant",
    llm=llm,
    tools=[db_tool],
    memory=memory,
    role=AGENT_ROLE,
    personality=AGENT_PERSONALITY,
)

# Example user query
result = dynamiq_name_assistant.run(
    input_data={
        "input": ("Show me the inventory"),
        "user_id": "1",
        "session_id": "1",
    },
    config=None,
)

# Print the result
print(result.output)
