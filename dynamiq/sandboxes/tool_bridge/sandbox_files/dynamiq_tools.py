"""Call the agent's own tools from code in the sandbox, while a sandbox-shell command runs.

The agent process runs each call, with credentials and per-run parameters that never enter the sandbox.

Python::

    from dynamiq_tools import ToolError, call, list_tools
    rows = call("warehouse-sql", {"query": "select 1"})

Shell::

    dynamiq-tools list
    dynamiq-tools get <tool>
    dynamiq-tools call <tool> --input '<json>'

Standard library only.
"""

import argparse
import json
import os
import socket
import sys

__all__ = ["ToolError", "call", "list_tools"]

DEFAULT_TIMEOUT = 300


class ToolError(Exception):
    """A call that did not succeed. ``type`` names the failure; a tool's own error message is not passed on."""

    def __init__(self, type_: str, reason: str | None = None):
        self.type = type_
        self.reason = reason
        super().__init__(f"{type_}: {reason}" if reason else type_)


def _request(payload: dict, timeout: float) -> dict:
    path, token = os.environ.get("DYNAMIQ_TOOLS_SOCKET"), os.environ.get("DYNAMIQ_TOOLS_TOKEN")
    if not path or not token:
        raise ToolError("BridgeUnavailable", "the agent's tools are reachable only during a sandbox-shell command")
    try:
        with socket.socket(socket.AF_UNIX) as s:
            s.connect(path)
            s.sendall((json.dumps({**payload, "token": token, "timeout": timeout}) + "\n").encode())
            line = s.makefile("rb").readline()
    except OSError as e:
        raise ToolError("BridgeUnavailable", f"cannot reach the bridge ({type(e).__name__})") from None
    if not line:
        raise ToolError("BridgeUnavailable", "the bridge closed the connection")
    reply = json.loads(line)
    if not reply.get("ok"):
        raise ToolError(reply.get("error", "BridgeError"), reply.get("reason"))
    return reply


def list_tools(timeout: float = 60) -> list[dict]:
    """The tools code may call, each with ``name``, ``description``, ``input_schema`` and ``app`` (MCP server)."""
    return _request({"op": "list"}, timeout)["result"]


def call(tool: str, input: dict | None = None, timeout: float = DEFAULT_TIMEOUT, **kwargs):  # noqa: A002
    """Run one of the agent's tools and return its result. Arguments go in ``input`` or as keywords."""
    return _request({"op": "call", "tool": tool, "input": {**(input or {}), **kwargs}}, timeout).get("result")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dynamiq-tools", description="Call the agent's tools.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List the tools.")
    commands.add_parser("get", help="Show a tool's input schema.").add_argument("tool")
    run = commands.add_parser("call", help="Call a tool and print its result.")
    run.add_argument("tool")
    run.add_argument("--input", default="{}", help="The arguments, as a JSON object.")
    run.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    args = parser.parse_args(argv)

    try:
        if args.command == "list":
            for tool in list_tools():
                print(f"{tool['name']}: {' '.join(tool['description'].split())[:160]}")
        elif args.command == "get":
            tool = next((t for t in list_tools() if t["name"] == args.tool), None)
            if tool is None:
                print(f"No tool named {args.tool!r}; see `dynamiq-tools list`.", file=sys.stderr)
                return 1
            print(json.dumps(tool, indent=2, ensure_ascii=False))
        else:
            try:
                arguments = json.loads(args.input)
            except json.JSONDecodeError as e:
                print(f"--input is not JSON: {e}", file=sys.stderr)
                return 1
            result = call(args.tool, arguments, timeout=args.timeout)
            print(result if isinstance(result, str) else json.dumps(result, indent=2, ensure_ascii=False))
    except ToolError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
