"""Content-free remote lifecycle authorization for hosted and embedded agents."""

from typing import Any, Literal

from pydantic import ConfigDict, Field, PrivateAttr, model_validator

from dynamiq.connections import Http
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.agents.hooks.core import (
    ALLOW,
    Block,
    BlockAs,
    Hook,
    HookContext,
    HookErrorPolicy,
    HookPoint,
    HookRunner,
    ToolCall,
    current_hook_ctx,
    register_hook,
)
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.remote_control import post_control_json, validate_control_connection


@register_hook("remote_lifecycle")
class RemoteLifecycleHook(Hook):
    """Authorize each model/tool call at a fixed HTTPS connection without sending its contents.

    The caller supplies opaque ``control_context`` through ``RunnableConfig.trusted_context``;
    model input, prompts, tool arguments and remote response reasons never enter the request or error.
    A denial, invalid response or transport failure always ends the run.
    """

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    connection: Http
    request_timeout_seconds: float = Field(default=5, gt=0, le=30)
    timeout_seconds: float = Field(default=10, gt=0, le=60)
    on_error: Literal[HookErrorPolicy.STOP] = HookErrorPolicy.STOP
    inherit: bool = True
    _client: Any = PrivateAttr(default=None)

    @model_validator(mode="after")
    def validate_endpoint(self):
        validate_control_connection(self.connection)
        return self

    def init_components(self, connection_manager: ConnectionManager | None = None) -> None:
        manager = connection_manager or ConnectionManager()
        self._client = manager.get_connection_client(self.connection)

    def _check(self, ctx: HookContext, phase: HookPoint):
        try:
            if self._client is None or ctx.config is None:
                raise ValueError("Lifecycle guard is not initialized")
            control = ctx.trusted.get("control_context")
            if not isinstance(control, dict):
                raise ValueError("Lifecycle control context is unavailable")
            payload = {
                "phase": phase.value,
                "run_id": ctx.config.run_id,
                "app_id": ctx.trusted.get("app_id"),
                "deployment_id": ctx.trusted.get("deployment_id"),
                "control_context": control,
            }
            result = post_control_json(self._client, self.connection, payload, self.request_timeout_seconds, 4096)
            if type(result.get("allow")) is not bool:
                raise ValueError("Lifecycle response is invalid")
        except Exception:
            # HTTP exceptions can contain endpoint credentials, headers and body text.
            raise RuntimeError("Remote lifecycle check failed") from None
        return ALLOW if result["allow"] else Block("Remote lifecycle authorization denied", BlockAs.FAIL)

    def before_model(self, ctx: HookContext, messages: list):
        return self._check(ctx, HookPoint.BEFORE_MODEL)

    def before_tool(self, ctx: HookContext, call: ToolCall):
        return self._check(ctx, HookPoint.BEFORE_TOOL)


def check_remote_model_lifecycle(config: RunnableConfig | None) -> None:
    """Guard a provider request without reapplying text-transform hooks."""
    ctx = current_hook_ctx.get()
    if ctx is None:
        return
    selected = [(key, hook) for key, hook in ctx.hooks if isinstance(hook, RemoteLifecycleHook)]
    if selected:
        HookRunner([hook for _, hook in selected], [key for key, _ in selected]).run(
            HookPoint.BEFORE_MODEL, ctx.for_call(config=config or ctx.config), []
        )


def check_remote_tool_lifecycle(config: RunnableConfig | None, name: str, tool_id: str) -> None:
    """Reauthorize a tool execution, including a node's automatic retry."""
    ctx = current_hook_ctx.get()
    if ctx is None:
        return
    selected = [(key, hook) for key, hook in ctx.hooks if isinstance(hook, RemoteLifecycleHook)]
    if selected:
        call = ToolCall(name=name, tool_id=tool_id, input={})
        HookRunner([hook for _, hook in selected], [key for key, _ in selected]).run(
            HookPoint.BEFORE_TOOL, ctx.for_call(config=config or ctx.config), {}, call=call
        )
