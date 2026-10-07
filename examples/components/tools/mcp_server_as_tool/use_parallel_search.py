"""Search and fetch with Dynamiq's MCP adapter, without a model or Parallel API key."""

import argparse
import json
from uuid import uuid4

from dynamiq.nodes.tools.mcp import MCPServer, MCPStreamableHTTP
from dynamiq.runnables import RunnableStatus


def run_example(query: str, url: str) -> dict:
    connection = MCPStreamableHTTP(
        url="https://search.parallel.ai/mcp",
        headers={"User-Agent": "Dynamiq-Parallel-MCP-Example (https://github.com/dynamiq-ai/dynamiq)"},
    )
    server = MCPServer(connection=connection, include_tools=["web_search", "web_fetch"])
    tools = {tool.name: tool for tool in server.get_mcp_tools()}
    # Keep one identifier across related calls, as recommended by Search MCP.
    session_id = str(uuid4())
    inputs = {
        "web_search": {"objective": query, "search_queries": [query], "session_id": session_id},
        "web_fetch": {"urls": [url], "search_queries": [query], "session_id": session_id},
    }
    output = {}
    for name, input_data in inputs.items():
        result = tools[name].run(input_data=input_data)
        if result.status != RunnableStatus.SUCCESS:
            raise RuntimeError(f"{name} failed: {result.error}")
        output[name] = result.output
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="Dynamiq MCP tool integration")
    parser.add_argument("--url", default="https://github.com/dynamiq-ai/dynamiq")
    args = parser.parse_args()
    print(json.dumps(run_example(args.query, args.url), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
