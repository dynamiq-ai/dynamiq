import os

from dynamiq import Workflow
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.tools.mcp_adapter import MCPServerAdapter, MPCConnection
from examples.llm_setup import setup_llm

llm = setup_llm()


def use_stdio_connection():
    """
    Initializes a Dynamiq workflow using an MCP server over stdio.

    The MCPAdapterTool retrieves all available tools from the server and makes them accessible to a ReActAgent,
    which can then use these tools to reason and respond to user queries.

    Returns:
        result (str): The result of executing the workflow.
    """
    stdio_connection = MPCConnection(
        command="python",
        args=[os.path.join("mcp_servers", "math_servers.py")],
    )

    mcp_tool_adapter = MCPServerAdapter(connection=stdio_connection)

    agent = ReActAgent(
        name="react-agent",
        id="react-agent",
        llm=llm,
        tools=[mcp_tool_adapter],
        max_loops=10,
    )

    wf = Workflow()
    wf.flow.add_nodes(agent)

    result = wf.run(input_data={"input": "What is the sum of 5 and 3? Use available tools to calculate the result."})

    print("Agent result:")
    print(result.output.get("react-agent", {}).get("output", {}).get("content"))
    return result


def use_sse_connection():
    """
    Initializes a workflow using an MCP server over SSE (Server-Sent Events).

    The MCPAdapterTool retrieves all available tools from the server and makes them accessible to a ReActAgent,
    which can then use these tools to reason and respond to user queries.


    Returns:
        result (str): The result of executing the workflow.
    """
    sse_connection = MPCConnection(url="http://localhost:8000/sse")

    mcp_tool_adapter = MCPServerAdapter(connection=sse_connection)

    agent = ReActAgent(
        name="react-agent",
        id="react-agent",
        llm=llm,
        tools=[mcp_tool_adapter],
        max_loops=10,
    )

    wf = Workflow()
    wf.flow.add_nodes(agent)

    result = wf.run(input_data={"input": "What is the current local time and weather in London?"})

    print("Agent result:")
    print(result.output.get("react-agent", {}).get("output", {}).get("content"))
    return result


if __name__ == "__main__":
    # Example of using stdio connection for local servers
    use_stdio_connection()

    # Example of using sse connection
    # Make sure to start the weather server before running this.
    # You can run it with: `python ./mcp_servers/weather_server.py`
    # It should be available at: http://localhost:8000/sse
    use_sse_connection()
