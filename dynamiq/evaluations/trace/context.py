"""Structured agent context for trace evaluation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from dynamiq.evaluations.trace.rendering import redact_secrets


class ToolSpec(BaseModel):
    name: str
    description: str = ""


class Message(BaseModel):
    role: str
    content: str


class AgentContext(BaseModel):
    """Structured bundle handed to the judge to ground suggestions."""

    agent_prompt: str = ""
    tools: list[ToolSpec] = Field(default_factory=list)
    conversation_history: list[Message] = Field(default_factory=list)

    @classmethod
    def from_agent(cls, agent: Any, history: list[Message] | None = None) -> AgentContext:
        tools: list[ToolSpec] = []
        for tool in getattr(agent, "tools", []) or []:
            tools.append(
                ToolSpec(
                    name=getattr(tool, "name", "") or "",
                    description=getattr(tool, "description", "") or "",
                )
            )
        return cls(
            agent_prompt=getattr(agent, "role", "") or "",
            tools=tools,
            conversation_history=list(history or []),
        )

    def render(self) -> str:
        lines: list[str] = [f"AGENT_PROMPT: {self.agent_prompt.strip()}", "TOOLS:"]
        if self.tools:
            for tool in self.tools:
                lines.append(f"  - name: {tool.name}")
                if tool.description:
                    lines.append(f"    description: {tool.description}")
        else:
            lines.append("  (none)")
        lines.append("CONVERSATION_HISTORY:")
        if self.conversation_history:
            for msg in self.conversation_history:
                lines.append(f"  [{msg.role}]: {msg.content}")
        else:
            lines.append("  (none)")
        return redact_secrets("\n".join(lines))
