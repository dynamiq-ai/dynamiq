"""A Python hook, referenced from dag.yaml with `type: custom_hooks.RequireNumericCustomerId`."""

from dynamiq.nodes.agents.hooks import ALLOW, Block, Decision, Hook, HookContext, ToolCall


class RequireNumericCustomerId(Hook):
    """Conditional block: refuse a call whose `customer_id` is not all digits.

    A hook is a pydantic config (so it round-trips through YAML) and must keep no per-run data on itself: one
    instance serves parallel tool calls and Map items. Per-run data goes in `ctx.hook_state`.
    """

    field: str = "customer_id"
    message: str = "customer_id must be numeric. Ask the user for a valid numeric customer id."

    def before_tool(self, ctx: HookContext, call: ToolCall) -> Decision:
        value = str(call.input.get(self.field, "")) if isinstance(call.input, dict) else ""
        return ALLOW if value.isdigit() else Block(self.message)
