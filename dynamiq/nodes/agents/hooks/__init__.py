from dynamiq.nodes.agents.hooks.builtin import (
    CallLimitHook,
    DetectorConfig,
    GuardHook,
    PIIHook,
    PromptInjectionHook,
    RegexHook,
    TextGuardHook,
    ToolPolicyHook,
    TransformHook,
)
from dynamiq.nodes.agents.hooks.core import (
    ALLOW,
    HOOK_REGISTRY,
    Allow,
    Ask,
    Block,
    BlockAs,
    Decision,
    Hook,
    HookContext,
    HookErrorPolicy,
    HookPoint,
    HookRun,
    HookRunner,
    LiveAnswerFilter,
    Modify,
    Skip,
    ToolCall,
    ToolResult,
    current_hook_ctx,
    register_hook,
    resolve_hook,
)

__all__ = [
    "hook_json_schemas",
    "ALLOW",
    "HOOK_REGISTRY",
    "Allow",
    "Ask",
    "Block",
    "BlockAs",
    "CallLimitHook",
    "Decision",
    "DetectorConfig",
    "GuardHook",
    "Hook",
    "HookContext",
    "HookErrorPolicy",
    "HookPoint",
    "HookRun",
    "HookRunner",
    "LiveAnswerFilter",
    "Modify",
    "PIIHook",
    "PromptInjectionHook",
    "RegexHook",
    "Skip",
    "TextGuardHook",
    "ToolCall",
    "ToolPolicyHook",
    "ToolResult",
    "current_hook_ctx",
    "TransformHook",
    "register_hook",
    "resolve_hook",
]


def hook_json_schemas() -> dict[str, dict]:
    """JSON schema of every built-in hook, keyed by its ``type`` (its import path): lets a UI render a form per hook
    type.

    Each schema describes the hook as it is written in a config, so it has the ``type`` property too, written as a
    node's is."""
    schemas = {}
    for hook_type, hook_class in sorted({cls.import_path(): cls for cls in HOOK_REGISTRY.values()}.items()):
        schema = hook_class.model_json_schema()
        schema["properties"] = {"type": {"type": "string", "enum": [hook_type]}, **schema.get("properties", {})}
        schema["required"] = ["type", *schema.get("required", [])]
        schemas[hook_type] = schema
    return schemas
