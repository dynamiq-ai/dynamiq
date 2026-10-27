from typing import TYPE_CHECKING

from dynamiq.utils.lazy_imports import lazy_exports

__all__ = [
    "Sandbox",
    "SandboxConfig",
    "SandboxConnectionError",
    "SandboxTool",
    "ShellCommandResult",
    "BedrockAgentCoreRuntimeSandbox",
    "BedrockAgentCoreSandbox",
    "DaytonaSandbox",
    "E2BSandbox",
    "E2BDesktopSandbox",
]

if TYPE_CHECKING:
    from .base import Sandbox, SandboxConfig, SandboxTool, ShellCommandResult
    from .bedrock_agentcore import BedrockAgentCoreSandbox
    from .bedrock_agentcore_runtime import BedrockAgentCoreRuntimeSandbox
    from .daytona import DaytonaSandbox
    from .e2b import E2BSandbox
    from .e2b_desktop import E2BDesktopSandbox
    from .exceptions import SandboxConnectionError

_LAZY_EXPORTS = {
    "Sandbox": ".base",
    "SandboxConfig": ".base",
    "SandboxTool": ".base",
    "ShellCommandResult": ".base",
    "BedrockAgentCoreRuntimeSandbox": ".bedrock_agentcore_runtime",
    "BedrockAgentCoreSandbox": ".bedrock_agentcore",
    "DaytonaSandbox": ".daytona",
    "E2BSandbox": ".e2b",
    "E2BDesktopSandbox": ".e2b_desktop",
    "SandboxConnectionError": ".exceptions",
}

lazy_exports(__name__, _LAZY_EXPORTS, globals())
