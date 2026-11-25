import base64
import re
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from dynamiq.checkpoints.types import RunPausedException
from dynamiq.nodes.agents.components.schema_generator import generate_function_calling_schemas
from dynamiq.nodes.agents.exceptions import BridgedApprovalRequired, HookBlockedException, ToolBlockedException
from dynamiq.nodes.agents.hooks.core import TOOL_POINTS, ToolCall
from dynamiq.nodes.schema_utils import strip_inaccessible_fields
from dynamiq.nodes.tools.agent_tool import SubAgentTool
from dynamiq.nodes.tools.context_manager import ContextManagerTool
from dynamiq.nodes.tools.human_feedback import HumanFeedbackTool
from dynamiq.nodes.tools.parallel_tool_calls import ParallelToolCallsTool
from dynamiq.nodes.tools.sql_executor import SQLExecutor
from dynamiq.nodes.tools.todo_tools import TodoWriteTool
from dynamiq.runnables import RunnableConfig
from dynamiq.sandboxes.base import Sandbox
from dynamiq.sandboxes.tool_bridge.sql_cli import sql_cli_connection, sql_cli_engine
from dynamiq.sandboxes.tool_bridge.types import in_bridged_call, refusal
from dynamiq.utils.logger import logger

if TYPE_CHECKING:
    from dynamiq.nodes import Node
    from dynamiq.nodes.agents.base import Agent

INTERNAL_TOOLS = (ContextManagerTool, ParallelToolCallsTool, TodoWriteTool)
AGENT_APP = "agent"  # the tools-cli app of tools that belong to no MCP server
TOOLS_ROUTE = re.compile(r"/v1/internal/connectors/([^/]+)/tools(?:/([^/]+))?(/call)?")
DATABASE_ROUTE = re.compile(r"/v1/internal/sql-cli/databases/([^/]+)(/credentials)?")


class AgentToolBridge:
    """Answers the requests of one shell command's code in the sandbox with the tools of ``agent``.

    A call takes the agent's tool path with the kwargs of its tool loop, so it gets the run's ``tool_params``, hooks
    and callbacks as a call of the model does, and shows in traces under the shell call. Code may call neither
    sub-agents, the agent's internal tools, the sandbox's own tools, tools that ask a person, nor calls a hook
    sends for approval. A failure crosses into the sandbox as its type only, since a tool's error text can carry
    what the sandbox must not see.

    A hook that ends the run on a call from code leaves its exception in ``run_ending``, for the agent to raise once
    the command returns, and every later request is refused.
    """

    def __init__(self, agent: "Agent", config: RunnableConfig | None, loop_kwargs: dict[str, Any]):
        self._agent = agent
        self._config = config
        self._kwargs = dict(loop_kwargs)
        self.run_ending: HookBlockedException | None = None

    def for_command(self, *, shell_run_id: Any) -> "AgentToolBridge":
        """The bridge for the command of the shell call ``shell_run_id``, under which its calls show in traces."""
        # run_id stays the agent's, so hook events of bridged calls land on its run as those of direct calls do.
        self._kwargs["parent_run_id"] = shell_run_id
        return self

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.run_ending is not None:
            return refusal("RunEnded", "a hook ended the run")
        op = request.get("op")
        if op == "list":
            return {"ok": True, "result": self.list_tools()}
        if op == "call":
            return self.call(request.get("tool"), request.get("input"))
        if op == "http":
            return self.http(request.get("method"), str(request.get("path") or ""), request.get("body"))
        return refusal("UnknownOperation")

    def _refusal_reason(self, tool: "Node") -> str | None:
        if isinstance(tool, SubAgentTool):
            return "agents cannot be called from code"
        if isinstance(tool, INTERNAL_TOOLS):
            return "the agent's internal tools cannot be called from code"
        if isinstance(tool, HumanFeedbackTool):
            return "code cannot ask a person"
        own_backend = self._agent.sandbox_backend
        if isinstance(getattr(tool, "sandbox", None), Sandbox) or (
            own_backend is not None and getattr(tool, "file_store", None) is own_backend
        ):
            return "the sandbox's own tools cannot be called from code; work in the sandbox directly"
        if tool.approval.enabled:
            return "the tool needs a person's approval, which code cannot ask for"
        return None

    def _bridgeable_tools(self) -> list[tuple[str, "Node"]]:
        tools = self._agent._runtime_tools
        names = self._agent._offered_tool_names(tools)
        return [(name, tool) for name, tool in zip(names, tools) if self._refusal_reason(tool) is None]

    def list_tools(self) -> list[dict[str, Any]]:
        entries = []
        for name, tool in self._bridgeable_tools():
            schema = generate_function_calling_schemas(
                [tool], False, self._agent.sanitize_tool_name, offered_names=[name]
            )[1]["function"]
            parameters = dict(schema.get("parameters") or {"type": "object", "properties": {}})
            parameters["properties"] = {k: v for k, v in (parameters.get("properties") or {}).items() if k != "thought"}
            parameters["required"] = [k for k in parameters.get("required") or [] if k != "thought"]
            owner = getattr(tool, "_owner_server", None)
            entries.append(
                {
                    "name": name,
                    "description": (tool.description or "").strip(),
                    "input_schema": parameters,
                    "app": getattr(owner, "name", None),
                }
            )
        return entries

    def call(self, name: Any, tool_input: Any) -> dict[str, Any]:
        if not isinstance(name, str) or not name:
            return refusal("InvalidRequest", "a tool name is required")
        tool = self._agent.tool_by_names.get(self._agent.sanitize_tool_name(name))
        if tool is None:
            return refusal("UnknownTool", "no such tool; list them with `dynamiq-tools list`")
        if (reason := self._refusal_reason(tool)) is not None:
            return refusal("ToolRefused", reason)
        if tool_input is None:
            tool_input = {}
        if not isinstance(tool_input, dict):
            return refusal("InvalidInput", "the input must be a JSON object")
        # As for the model's calls, fields hidden from it (mcp_http_headers) take their values from tool_params only.
        tool_input, stripped = strip_inaccessible_fields(tool.resolved_input_schema, tool_input)
        if stripped:
            logger.warning(f"Tool bridge: dropped {stripped} from the input of '{tool.name}' sent from the sandbox.")

        token = in_bridged_call.set(True)
        try:
            content, _files, _meta = self._agent._run_tool(
                tool,
                tool_input,
                self._config,
                update_run_depends=False,
                is_parallel=True,
                tool_run_id=str(uuid4()),
                raw_output=True,
                **self._kwargs,
            )
        except BridgedApprovalRequired:
            return refusal("ApprovalRequired", "the call needs a person's approval, which code cannot ask for")
        except ToolBlockedException:
            return refusal("ToolBlocked")
        except HookBlockedException as e:
            self.run_ending = self.run_ending or e
            return refusal("RunEnded", "a hook ended the run")
        except RunPausedException:
            logger.warning(f"Tool bridge: '{tool.name}' tried to pause the run from code; refused.")
            return refusal("PauseNotAllowed", "a call from code cannot pause the run")
        except Exception as e:
            logger.warning(f"Tool bridge: call of '{tool.name}' from the sandbox failed with {type(e).__name__}.")
            return refusal(type(e).__name__)
        finally:
            in_bridged_call.reset(token)
        return {"ok": True, "result": _jsonable(content)}

    def http(self, method: Any, path: str, body: Any) -> dict[str, Any]:
        """Answers the platform routes the template's CLIs call, in the platform's shapes (``{"data": ...}``).

        tools-cli: ``/v1/internal/connectors[/{app}/tools[/{tool}[/call]]]``, an app being a tool's MCP server or
        ``agent``. sql-cli: ``/v1/internal/sql-cli/databases[/{slug}[/credentials]]``, the agent's SQL tools.
        """
        path = path.rstrip("/")
        if method == "GET" and path == "/v1/internal/connectors":
            apps = sorted({tool["app"] or AGENT_APP for tool in self.list_tools()})
            return _ok([{"slug": app, "name": app, "description": None} for app in apps])
        if match := TOOLS_ROUTE.fullmatch(path):
            app, name, is_call = match.group(1), match.group(2), bool(match.group(3))
            tools = [tool for tool in self.list_tools() if (tool["app"] or AGENT_APP) == app]
            if method == "GET" and name is None:
                return _ok([{"name": tool["name"], "description": tool["description"]} for tool in tools])
            tool = next((tool for tool in tools if tool["name"] == name), None)
            if tool is None:
                return _http_error(404, "not_found", f"Tool '{name}' not found.")
            if method == "GET" and not is_call:
                return _ok(tool)
            if method == "POST" and is_call:
                tool_input = body.get("input") if isinstance(body, dict) else None
                reply = self.call(name, tool_input)
                if reply["ok"]:
                    return _ok(reply["result"])
                return _http_error(422, reply["error"], reply.get("reason") or reply["error"])
        if method == "GET" and path == "/v1/internal/sql-cli/databases":
            return _ok([entry for entry, _ in self._databases().values()])
        if method == "GET" and (match := DATABASE_ROUTE.fullmatch(path)):
            found = self._databases().get(match.group(1))
            if found is None:
                return _http_error(404, "not_found", f"Database '{match.group(1)}' not found.")
            entry, tool = found
            return self._database_credentials(entry, tool) if match.group(2) else _ok(entry)
        return _http_error(404, "not_found", "Not found.")

    def _databases(self) -> dict[str, tuple[dict[str, Any], SQLExecutor]]:
        """The databases of the agent's SQL tools that sql-cli can use, by slug."""
        found: dict[str, tuple[dict[str, Any], SQLExecutor]] = {}
        for name, tool in self._bridgeable_tools():
            if not isinstance(tool, SQLExecutor) or (engine := sql_cli_engine(tool.connection)) is None:
                continue
            slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "database"
            if slug in found:
                slug = f"{slug}-{tool.id[:8]}"
            entry = {
                "id": tool.id,
                "slug": slug,
                "name": tool.name,
                "description": (tool.description or "").strip() or None,
                "engine": engine,
            }
            found[slug] = (entry, tool)
        return found

    def _database_credentials(self, entry: dict[str, Any], tool: SQLExecutor) -> dict[str, Any]:
        """The credentials sql-cli connects with, which any code in the command can then read.

        Served only where the sandbox allows CLI credentials and no hook governs the tool, since a direct connection
        bypasses the checks and masking hooks apply to each call. Each grant shows in traces as a run of the tool.
        """
        if not self._agent.sandbox_backend.tool_bridge.cli_credentials_allowed:
            message = f"This agent's sandbox gets no database credentials; query through the '{tool.name}' tool."
            return _http_error(403, "forbidden", message)
        call = ToolCall(name=tool.name, input={}, tool_id=tool.authored_id)
        if any(hook.points() & set(TOOL_POINTS) and hook.applies_to(call) for hook in self._agent._hook_runner().hooks):
            message = f"Hooks check every call of '{tool.name}', which a direct connection would bypass; use the tool."
            return _http_error(403, "forbidden", message)
        try:
            connection = sql_cli_connection(tool.connection)
        except ConnectionError as e:
            logger.warning(f"Tool bridge: no credentials for database '{entry['slug']}' ({type(e).__name__}).")
            return _http_error(502, "bad_gateway", f"The credentials of '{tool.name}' are unavailable.")
        callbacks = self._config.callbacks if self._config else []
        run_kwargs = {"run_id": uuid4(), "parent_run_id": self._kwargs["parent_run_id"]}
        tool.run_on_node_start(callbacks, {"credentials_for": "sql-cli"}, **run_kwargs)
        tool.run_on_node_end(callbacks, {"content": "Credentials served to sql-cli."}, **run_kwargs)
        logger.info(f"Tool bridge: served the credentials of database '{entry['slug']}' to sql-cli.")
        return _ok({"engine": entry["engine"], "connection": connection})


def _ok(data: Any) -> dict[str, Any]:
    return {"ok": True, "status": 200, "body": {"data": data}}


def _http_error(status: int, code: str, message: str) -> dict[str, Any]:
    return {"ok": True, "status": status, "body": {"error": {"code": code, "message": message}}}


def _jsonable(content: Any) -> Any:
    if isinstance(content, (bytes, bytearray)):
        try:
            return bytes(content).decode("utf-8")
        except UnicodeDecodeError:
            return {"base64": base64.b64encode(bytes(content)).decode()}
    if hasattr(content, "model_dump"):
        return content.model_dump(mode="json")
    return content
