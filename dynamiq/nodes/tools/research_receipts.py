"""Bounded observations emitted by web transports, separately from model-facing output."""

from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Field

from dynamiq.runnables import RunnableConfig


class ResearchReceipt(BaseModel):
    kind: Literal["search", "fetch", "pdf"]
    provider: Literal["exa", "jina", "firecrawl"]
    status: Literal["succeeded", "failed"]
    query: str | None = Field(default=None, max_length=4096)
    url: str | None = Field(default=None, max_length=4096)
    final_url: str | None = Field(default=None, max_length=4096)
    result_urls: list[str] = Field(default_factory=list, max_length=50)
    status_code: int | None = None
    error_code: Literal["transport_error", "empty_content", "provider_error"] | None = None
    # These APIs do not report a monetary price per request. Consumers must preserve that uncertainty.
    cost_usd: str | None = None


def receipt_url(value: Any) -> str | None:
    if not isinstance(value, str) or len(value) > 4096:
        return None
    try:
        parsed = urlsplit(value)
        if parsed.scheme in {"https", "http"} and parsed.hostname and not parsed.username and not parsed.password:
            return value
    except ValueError:
        pass
    return None


def emit_research_receipt(node: Any, config: RunnableConfig, kwargs: dict, **values: Any) -> None:
    """Emit only transport observations, never document bodies, headers or exception text."""
    values["url"] = receipt_url(values.get("url"))
    values["final_url"] = receipt_url(values.get("final_url"))
    values["result_urls"] = [
        clean for value in values.get("result_urls", [])[:50] if (clean := receipt_url(value)) is not None
    ]
    if isinstance(values.get("query"), str):
        values["query"] = values["query"][:4096]
    if type(values.get("status_code")) is not int:
        values["status_code"] = None
    receipt = ResearchReceipt(**values)
    node.run_on_node_execute_run(config.callbacks, research_receipt=receipt.model_dump(exclude_none=True), **kwargs)
