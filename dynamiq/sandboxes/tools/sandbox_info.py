import logging
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.node import ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.sandboxes.base import Sandbox, SandboxInfo

logger = logging.getLogger(__name__)


class SandboxInfoInputSchema(BaseModel):
    """Input schema for SandboxInfoTool."""

    port: int | None = Field(
        default=None,
        description="Optional port number. If provided and the sandbox supports it, "
        "the response includes the public URL to access a service running on this port (e.g. dev server).",
    )


class SandboxInfoTool(Node):
    """A tool for the agent to get sandbox metadata and, when needed, the public URL for a port.

    Use this when the agent starts a server in the sandbox (e.g. `npm run dev` on port 5173)
    so it can report the shareable URL to the user. The sandbox backend (e.g. E2B) exposes
    services at a public URL; this tool returns that URL for a given port.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "SandboxInfoTool"
    description: str = (
        "Get information about the current sandbox (paths, sandbox id, and optional public URL).\n\n"
        "Call this when you need to report a URL to the user for a service you started in the sandbox "
        "(e.g. a dev server). Pass the port number the service actually listens on (check the server output: "
        "e.g. Vite -> 5173, Next.js -> 3000, python -m http.server -> 8000). The URL will only work if a "
        "process is listening on that port in the sandbox.\n\n"
        "Parameters:\n"
        "- port (int, optional): Port the service listens on. If provided, response includes "
        "public_host and public_url (https) so the user can open the app in a browser.\n\n"
        "Examples:\n"
        '- {"port": 5173}  → get sandbox info and public URL for port 5173 (Vite default)\n'
        "- {}  → get base_path, sandbox_id only"
    )

    sandbox: Sandbox = Field(..., description="Sandbox backend to query.")
    model_config = ConfigDict(arbitrary_types_allowed=True)
    input_schema: ClassVar[type[SandboxInfoInputSchema]] = SandboxInfoInputSchema

    @property
    def to_dict_exclude_params(self) -> set[str]:
        return super().to_dict_exclude_params | {"sandbox": True}

    def to_dict(self, **kwargs) -> dict[str, Any]:
        for_tracing = kwargs.pop("for_tracing", False)
        data = super().to_dict(for_tracing=for_tracing, **kwargs)
        data["sandbox"] = self.sandbox.to_dict(for_tracing=for_tracing, **kwargs) if self.sandbox else None
        return data

    def execute(
        self,
        input_data: SandboxInfoInputSchema,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Return sandbox info and optional public URL for the given port."""
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        info: SandboxInfo = self.sandbox.get_sandbox_info(port=input_data.port)

        if input_data.port is not None:
            if info.public_url:
                logger.info(
                    "SandboxInfoTool: port=%s -> public_url=%s "
                    "(ensure the server is running on this port in the sandbox)",
                    input_data.port,
                    info.public_url,
                )
            elif info.public_url_error:
                logger.warning(
                    "SandboxInfoTool: port=%s -> error: %s",
                    input_data.port,
                    info.public_url_error,
                )
        else:
            logger.info(
                "SandboxInfoTool: sandbox_id=%s base_path=%s",
                info.sandbox_id,
                info.base_path,
            )

        lines = [
            f"base_path: {info.base_path}",
        ]
        if info.sandbox_id:
            lines.append(f"sandbox_id: {info.sandbox_id}")
        if input_data.port is not None:
            if info.public_url:
                lines.append(f"public_url: {info.public_url}")
                lines.append("Share this URL with the user to open the app in a browser.")
            elif info.public_url_error:
                lines.append(f"public_url_error: {info.public_url_error}")

        content = "\n".join(lines)
        return {"content": content, "sandbox_info": info.model_dump()}
