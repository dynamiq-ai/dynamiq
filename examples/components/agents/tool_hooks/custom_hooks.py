"""Python tool hooks referenced from dag.yaml with `type: custom_hooks.<Class>`."""

from typing import Any

from dynamiq.nodes.agents.hooks import ToolHook


class RequireNumericCustomerId(ToolHook):
    """Conditional block: refuse a call whose `customer_id` is not all digits."""

    field: str = "customer_id"

    def should_block(self, tool_name: str, tool_input: Any) -> bool:
        value = str(tool_input.get(self.field, "")) if isinstance(tool_input, dict) else ""
        return not value.isdigit()
