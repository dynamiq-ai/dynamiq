"""Declarative and Python before/after hooks for agent tool calls.

A ``ToolHook`` is a plain pydantic config (like ``ApprovalConfig`` / ``MockConfig``), so it is dumped to and
loaded from YAML together with the agent. It reuses the node ``InputTransformer`` / ``OutputTransformer``
(JSONPath ``path`` / ``selector``) to rewrite what a tool receives and what it returns, and can veto a call.

For custom logic subclass ``ToolHook`` and override ``should_block`` / ``apply_before`` / ``apply_after``; in YAML
reference the subclass with ``type: my_pkg.hooks.MyHook``.

A hook that raises is handled according to ``on_error`` (fail closed by default).

Hooks are not applied to ``ContextManagerTool`` or sub-agent tools.
"""

import importlib
import json
import re
from enum import Enum
from typing import Any, ClassVar

from pydantic import BaseModel, Field, model_serializer, model_validator

from dynamiq.nodes.agents.exceptions import ToolExecutionException, ToolHookStopException
from dynamiq.nodes.node import InputTransformer, Node, OutputTransformer, Transformer
from dynamiq.utils.jsonpath import _is_rooted, is_jsonpath
from dynamiq.utils.logger import logger

TRACE_PREVIEW_CHARS = 500


def _sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "", name.replace(" ", "-"))


class HookErrorPolicy(str, Enum):
    """What to do when a hook itself raises.

    Attributes:
        BLOCK: Fail closed. The tool call is aborted (before hook) or its result withheld (after hook) and the
            LLM gets an observation. Default: a crashing guard or redaction step must not let raw data through.
        STOP: End the agent run.
        SKIP: Fail open. Log the error and continue with the unmodified input/result.
    """

    BLOCK = "block"
    STOP = "stop"
    SKIP = "skip"


def _check_transformer(transformer: Transformer, label: str) -> None:
    """Reject expressions that would only fail (or silently become literals) in the middle of a run."""
    if transformer.path and not is_jsonpath(transformer.path):
        raise ValueError(f"{label}.path {transformer.path!r} is not a valid JSONPath expression")
    for key, expression in (transformer.selector or {}).items():
        if _is_rooted(expression) and not is_jsonpath(expression):
            raise ValueError(f"{label}.selector[{key!r}] {expression!r} is not a valid JSONPath expression")


class HookBase(BaseModel):
    """Shared base of tool and model hooks: dumps the class path of subclasses so YAML can load them back."""

    @model_serializer(mode="wrap")
    def _serialize_with_type(self, handler):
        data = handler(self)
        if not type(self).__dict__.get("_is_base_hook", False):
            data = {"type": f"{type(self).__module__}.{type(self).__qualname__}", **data}
        return data


class ToolHook(HookBase):
    """Before/after hook for agent tool calls.

    Attributes:
        tools: Names of the tools the hook applies to. Empty means every tool.
        input_transformer: Applied to the tool input dict before the tool runs.
        output_transformer: Applied to ``{"content": <tool result>}`` after the tool succeeds.
        block: Veto the call before the tool runs.
        block_message: Message returned to the LLM as the observation (or carried by the stop exception).
        stop_agent: With ``block``, end the agent run instead of returning ``block_message`` to the LLM.
        on_error: What to do if the hook itself raises.
    """

    tools: list[str] = Field(default_factory=list, description="Tool names to match. Empty means all tools.")
    input_transformer: InputTransformer = Field(default_factory=InputTransformer)
    output_transformer: OutputTransformer = Field(default_factory=OutputTransformer)
    block: bool = False
    block_message: str = "Tool call blocked by hook."
    stop_agent: bool = False
    on_error: HookErrorPolicy = HookErrorPolicy.BLOCK

    _is_base_hook: ClassVar[bool] = True

    @model_validator(mode="after")
    def _validate_transformers(self):
        _check_transformer(self.input_transformer, "input_transformer")
        _check_transformer(self.output_transformer, "output_transformer")
        return self

    def matches(self, tool_name: str) -> bool:
        if not self.tools:
            return True
        return _sanitize(tool_name) in {_sanitize(name) for name in self.tools}

    def should_block(self, tool_name: str, tool_input: Any) -> bool:
        """Whether to veto this call. Override for conditional blocking."""
        return self.block

    def apply_before(self, tool_name: str, tool_input: Any) -> Any:
        """Rewrite the tool input. Override for custom logic."""
        if isinstance(tool_input, dict) and (self.input_transformer.path or self.input_transformer.selector):
            return Node.transform(tool_input, self.input_transformer)
        return tool_input

    def apply_after(self, tool_name: str, tool_result: Any) -> Any:
        """Rewrite the tool result (exposed to the transformer as ``{"content": result}``).

        A string result holding a JSON object/array is parsed first, so JSONPath can select fields from it;
        a non-string outcome is serialized back to JSON. Override for custom logic.
        """
        if not (self.output_transformer.path or self.output_transformer.selector):
            return tool_result

        data, parsed = tool_result, False
        if isinstance(tool_result, str):
            try:
                loaded = json.loads(tool_result)
            except ValueError:
                loaded = None
            if isinstance(loaded, (dict, list)):
                data, parsed = loaded, True

        transformed = Node.transform({"content": data}, self.output_transformer)
        if not isinstance(transformed, dict) or "content" not in transformed:
            logger.warning(f"Tool hook for '{tool_name}': output_transformer produced no 'content'; result kept.")
            return tool_result

        content = transformed["content"]
        if content is None:
            logger.warning(f"Tool hook for '{tool_name}': output_transformer matched nothing; 'content' is None.")
        if parsed and not isinstance(content, str) and content is not None:
            content = json.dumps(content, ensure_ascii=False)
        return content


def resolve_hook(item: Any, base_class: type[HookBase]) -> Any:
    """Turn a dict with ``type: pkg.module.Class`` into that ``base_class`` subclass; leave everything else as is."""
    if isinstance(item, dict) and item.get("type"):
        module_name, _, class_name = str(item["type"]).rpartition(".")
        try:
            hook_class = getattr(importlib.import_module(module_name), class_name)
        except (ImportError, AttributeError, ValueError) as e:
            raise ValueError(f"Hook type {item['type']!r} could not be imported: {e}") from e
        if not (isinstance(hook_class, type) and issubclass(hook_class, base_class)):
            raise ValueError(f"Hook type {item['type']!r} is not a {base_class.__name__} subclass")
        return hook_class.model_validate({k: v for k, v in item.items() if k != "type"})
    return item


def resolve_tool_hook(item: Any) -> Any:
    return resolve_hook(item, ToolHook)


def _preview(value: Any, limit: int = TRACE_PREVIEW_CHARS) -> str:
    text = value if isinstance(value, str) else repr(value)
    return text if len(text) <= limit else f"{text[:limit]}... [{len(text)} chars]"


def _record(trace: list[dict] | None, **entry: Any) -> None:
    if trace is not None:
        trace.append(entry)


def _run_hook(hook: ToolHook, index: int, phase: str, tool_name: str, fn, fallback: Any, trace: list[dict] | None):
    """Run one hook step; a crash is handled per ``hook.on_error`` (deliberate hook exceptions pass through)."""
    try:
        return fn()
    except (ToolExecutionException, ToolHookStopException):
        raise
    except Exception as e:
        policy = hook.on_error
        detail = f"{type(e).__name__}: {e}"
        logger.error(f"Tool hook[{index}] {phase} for '{tool_name}' failed ({detail}); on_error={policy.value}")
        _record(trace, tool=tool_name, phase="error", hook=index, step=phase, policy=policy.value, error=detail)
        if policy == HookErrorPolicy.SKIP:
            return fallback
        message = f"Tool hook[{index}] {phase} for '{tool_name}' failed: {e}"
        if policy == HookErrorPolicy.STOP:
            raise ToolHookStopException(message) from e
        raise ToolExecutionException(
            f"{message} ({'tool result withheld' if phase == 'output' else 'call not executed'})"
        ) from e


def apply_before_tool_hooks(
    hooks: list[ToolHook], tool_name: str, tool_input: Any, trace: list[dict] | None = None
) -> Any:
    """Run matching hooks in list order and return the (possibly rewritten) tool input.

    Every applied rewrite is logged and appended to ``trace`` (tracing metadata).
    """
    for index, hook in enumerate(hooks):
        if not hook.matches(tool_name):
            continue
        rewritten = _run_hook(
            hook, index, "input", tool_name, lambda h=hook: h.apply_before(tool_name, tool_input), tool_input, trace
        )
        if rewritten is not tool_input:
            logger.info(f"Tool hook[{index}] input for '{tool_name}': {_preview(tool_input)} -> {_preview(rewritten)}")
            _record(
                trace,
                tool=tool_name,
                phase="input",
                hook=index,
                changed=rewritten != tool_input,
                before=_preview(tool_input),
                after=_preview(rewritten),
            )
            tool_input = rewritten
    return tool_input


def check_tool_hooks_block(
    hooks: list[ToolHook], tool_name: str, tool_input: Any = None, trace: list[dict] | None = None
) -> None:
    """Raise if any matching hook blocks the call (logged and traced before raising)."""
    for index, hook in enumerate(hooks):
        if not hook.matches(tool_name):
            continue
        blocked = _run_hook(
            hook, index, "block", tool_name, lambda h=hook: h.should_block(tool_name, tool_input), False, trace
        )
        if not blocked:
            continue
        logger.warning(
            f"Tool hook[{index}] BLOCKED '{tool_name}' "
            f"({'stopping the agent' if hook.stop_agent else 'message returned to the LLM'}): {hook.block_message}"
        )
        _record(
            trace, tool=tool_name, phase="block", hook=index, stop_agent=hook.stop_agent, message=hook.block_message
        )
        if hook.stop_agent:
            raise ToolHookStopException(hook.block_message)
        raise ToolExecutionException(hook.block_message)


def apply_after_tool_hooks(
    hooks: list[ToolHook], tool_name: str, tool_result: Any, trace: list[dict] | None = None
) -> Any:
    """Run matching hooks in reverse list order and return the (possibly rewritten) tool result.

    The original result is deliberately not logged or traced (an output hook is often a redaction step);
    only its size and the rewritten result are.
    """
    for index in range(len(hooks) - 1, -1, -1):
        hook = hooks[index]
        if not hook.matches(tool_name):
            continue
        rewritten = _run_hook(
            hook, index, "output", tool_name, lambda h=hook: h.apply_after(tool_name, tool_result), tool_result, trace
        )
        if rewritten is not tool_result:
            before_chars = len(tool_result) if isinstance(tool_result, str) else len(repr(tool_result))
            logger.info(f"Tool hook[{index}] output for '{tool_name}': {before_chars} chars -> {_preview(rewritten)}")
            _record(
                trace,
                tool=tool_name,
                phase="output",
                hook=index,
                changed=rewritten != tool_result,
                before_chars=before_chars,
                after=_preview(rewritten),
            )
            tool_result = rewritten
    return tool_result
