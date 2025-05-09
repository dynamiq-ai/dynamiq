import os

from dynamiq import Workflow
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.tools.mcp import MCPServer, MCPSse, MCPStdio, MCPStreamableHTTP
from examples.llm_setup import setup_llm

llm = setup_llm()


def setup_wf(connection: MCPSse | MCPStdio | MCPStreamableHTTP):
    mcp_server = MCPServer(connection=connection)

    agent = ReActAgent(
        name="react-agent",
        id="react-agent",
        llm=llm,
        tools=[mcp_server],
        max_loops=10,
    )

    wf = Workflow()
    wf.flow.add_nodes(agent)
    return wf


def use_stdio_connection(path_to_server: str):
    """
    Initializes a Dynamiq workflow using an MCP server over stdio.

    The MCPTool retrieves all available tools from the server and makes them accessible to a ReActAgent,
    which can then use these tools to reason and respond to user queries.

    Args:
        path_to_server (str): Path to the MCP server.

    Returns:
        result (str): The result of executing the workflow.
    """
    stdio_connection = MCPStdio(
        command="python",
        args=[path_to_server],
    )

    wf = setup_wf(stdio_connection)
    result = wf.run(input_data={"input": "What is the sum of 5 and 3? Use available tools to calculate the result."})

    print("Agent result:")
    print(result.output.get("react-agent", {}).get("output", {}).get("content"))
    return result


def use_sse_connection():
    """
    Initializes a workflow using an MCP server over SSE (Server-Sent Events).

    Returns:
        result (str): The result of executing the workflow.
    """
    sse_connection = MCPSse(url="http://localhost:8000/sse")

    wf = setup_wf(sse_connection)
    result = wf.run(input_data={"input": "What is the current local time and weather in London?"})

    print("Agent result:")
    print(result.output.get("react-agent", {}).get("output", {}).get("content"))
    return result


def use_streamablehttp_connection():
    """
    Initializes a workflow using an MCP server over streamable HTTP

    Returns:
        result (str): The result of executing the workflow.
    """
    streamablehttp_connection = MCPStreamableHTTP(url="http://localhost:8000/mcp")

    wf = setup_wf(streamablehttp_connection)
    result = wf.run(input_data={"input": "What is the current local time and weather in London?"})

    print("Agent result:")
    print(result.output.get("react-agent", {}).get("output", {}).get("content"))
    return result


def use_remote_server_oauth():
    """
    Connects to a remote MCP server using OAuth via a subprocess with stdio.

    Returns:
        result (str): The result of executing the workflow.
    """
    stdio_connection = MCPStdio(
        command="npx",
        args=["-y", "mcp-remote", "https://mcp.linear.app/sse"],
    )

    wf = setup_wf(stdio_connection)
    result = wf.run(input_data={"input": "List all users"})

    print("Agent result:")
    print(result.output.get("react-agent", {}).get("output", {}).get("content"))
    return result


def use_remote_server_open():
    """
    Connects to an open remote MCP server using a subprocess with stdio, without requiring authentication.

    Returns:
        result (str): The result of executing the workflow.
    """
    stdio_connection = MCPStdio(command="npx", args=["mcp-remote", "https://remote.mcpservers.org/fetch/mcp"])

    wf = setup_wf(stdio_connection)
    result = wf.run(input_data={"input": "Retrieve the information displayed on https://www.apple.com/ page"})

    print("Agent result:")
    print(result.output.get("react-agent", {}).get("output", {}).get("content"))
    return result


def use_local_server_with_token():
    """
    Runs a local MCP server using an access token via NPX subprocess.

    Returns:
        result (str): The result of executing the workflow.
    """
    stdio_connection = MCPStdio(command="npx", args=["-y", "tavily-mcp@0.1.4"], env={"TAVILY_API_KEY": "tvly-token"})

    wf = setup_wf(stdio_connection)
    result = wf.run(input_data={"input": "Retrieve latest news from Apple Inc."})

    print("Agent result:")
    print(result.output.get("react-agent", {}).get("output", {}).get("content"))
    return result


if __name__ == "__main__":
    # Example of using Stdio connection for local servers
    path_to_server = os.path.join("mcp_servers", "math_server.py")
    use_stdio_connection(path_to_server)

    # Example of using SSE connection
    # Make sure to start the weather server before running this.
    # You can run it with:
    # `python examples/components/tools/mcp_server_as_tool/mcp_servers/weather_server.py --transport sse`
    # It should be available at: http://localhost:8000/sse
    use_sse_connection()

    # Example of using streamable HTTP connection
    # Make sure to start the weather server before running this.
    # You can run it with:
    # `python examples/components/tools/mcp_server_as_tool/mcp_servers/weather_server.py --transport streamable-http`
    # It should be available at: http://localhost:8000/mcp
    use_streamablehttp_connection()

    # Connecting to remote server using oauth
    use_remote_server_oauth()

    # Connecting to remote open server
    use_remote_server_open()

    # Running local server using npx and access token
    use_local_server_with_token()
