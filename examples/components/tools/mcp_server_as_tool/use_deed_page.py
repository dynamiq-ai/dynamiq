"""Publish an HTML page with Dynamiq's MCP adapter, without an LLM or deed.page account."""

import argparse
import json
import os
from pathlib import Path

from dynamiq.nodes.tools.mcp import MCPServer, MCPStreamableHTTP
from dynamiq.runnables import RunnableStatus

SAMPLE_PAGE = (
    "<!doctype html><meta charset='utf-8'><title>Dynamiq report</title>"
    "<h1>Dynamiq report</h1><p>Published through the deed.page MCP example.</p>"
)


def run_tool(name: str, input_data: dict) -> dict:
    connection = MCPStreamableHTTP(
        url="https://deed.page/mcp",
        headers={"User-Agent": "Dynamiq-deed.page-MCP-Example (https://github.com/dynamiq-ai/dynamiq)"},
    )
    server = MCPServer(connection=connection, include_tools=["publish_site", "delete_site"])
    tools = {tool.name: tool for tool in server.get_mcp_tools()}
    result = tools[name].run(input_data=input_data)
    if result.status != RunnableStatus.SUCCESS:
        raise RuntimeError(f"{name} failed: {result.error}")
    return result.output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, help="HTML file to publish as index.html (default: a sample page)")
    parser.add_argument("--slug", help="Optional subdomain to claim, e.g. my-team-report")
    parser.add_argument("--ttl", type=int, default=3600, help="Seconds until the page expires; 0 keeps it")
    parser.add_argument("--delete", metavar="SLUG", help="Delete a page using the token in DEED_PAGE_TOKEN")
    args = parser.parse_args()
    if args.delete:
        token = os.environ.get("DEED_PAGE_TOKEN")
        if not token:
            parser.error("--delete needs the owner token from the first publish in DEED_PAGE_TOKEN")
        output = run_tool("delete_site", {"slug": args.delete, "token": token})
    else:
        if args.ttl < 0:
            parser.error("--ttl must be 0 or more")
        html = args.file.read_text(encoding="utf-8") if args.file else SAMPLE_PAGE
        input_data = {"files": {"index.html": html}}
        if args.slug:
            input_data["slug"] = args.slug
        if args.ttl:
            input_data["ttl_seconds"] = args.ttl
        output = run_tool("publish_site", input_data)
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
