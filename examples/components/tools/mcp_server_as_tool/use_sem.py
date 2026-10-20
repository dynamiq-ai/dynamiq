"""Look up local code through Dynamiq's MCP adapter, without an LLM."""

import argparse
import json
from pathlib import Path

from dynamiq.nodes.tools.mcp import MCPServer, MCPStdio
from dynamiq.runnables import RunnableStatus


def run_example(repo: str, query: str) -> dict:
    checkout = Path(repo).expanduser().resolve(strict=True)
    if not checkout.is_dir():
        raise ValueError("repo must be a directory")
    server = MCPServer(
        connection=MCPStdio(command="sem", args=["mcp"], env={"SEM_REPO": str(checkout)}),
        include_tools=["sem_find", "sem_grep"],
    )
    tools = {tool.name: tool for tool in server.get_mcp_tools()}
    result = tools["sem_find"].run(input_data={"query": query, "mode": "context", "token_budget": 1200})
    if result.status != RunnableStatus.SUCCESS:
        raise RuntimeError(f"sem_find failed: {result.error}")
    return result.output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="Local checkout to search")
    parser.add_argument("--query", required=True, help="Symbol name or code query")
    args = parser.parse_args()
    print(json.dumps(run_example(args.repo, args.query), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
