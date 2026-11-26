import base64
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from dynamiq.checkpoints.types import RunPausedException
from dynamiq.nodes.agents.components.schema_generator import generate_function_calling_schemas
from dynamiq.nodes.agents.exceptions import BridgedApprovalRequired, HookBlockedException, ToolBlockedException
from dynamiq.nodes.schema_utils import strip_inaccessible_fields
from dynamiq.nodes.tools.agent_tool import SubAgentTool
from dynamiq.nodes.tools.context_manager import ContextManagerTool
from dynamiq.nodes.tools.human_feedback import HumanFeedbackTool
from dynamiq.nodes.tools.parallel_tool_calls import ParallelToolCallsTool
from dynamiq.nodes.tools.todo_tools import TodoWriteTool
from dynamiq.runnables import RunnableConfig
from dynamiq.sandboxes.base import Sandbox
from dynamiq.sandboxes.tool_bridge.types import in_bridged_call, refusal
from dynamiq.utils.logger import logger

if TYPE_CHECKING:
    from dynamiq.nodes import Node
    from dynamiq.nodes.agents.base import Agent

INTERNAL_TOOLS = (ContextManagerTool, ParallelToolCallsTool, TodoWriteTool)


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


def _jsonable(content: Any) -> Any:
    if isinstance(content, (bytes, bytearray)):
        try:
            return bytes(content).decode("utf-8")
        except UnicodeDecodeError:
            return {"base64": base64.b64encode(bytes(content)).decode()}
    if hasattr(content, "model_dump"):
        return content.model_dump(mode="json")
    return content
