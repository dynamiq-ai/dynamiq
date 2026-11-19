"""Bounded observations emitted by web transports, separately from model-facing output."""

from decimal import Decimal, InvalidOperation
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, BeforeValidator, Field

from dynamiq.runnables import RunnableConfig

MAX_COST_REPRESENTATION = 64


def receipt_units(value: Any) -> int | None:
    return value if type(value) is int and 0 <= value <= 2**63 - 1 else None


def receipt_header_units(value: Any) -> int | None:
    if isinstance(value, str):
        digits = value.strip()
        if digits and digits.isascii() and digits.isdecimal():
            digits = digits.lstrip("0") or "0"
            value = int(digits) if len(digits) <= 19 else None
    return receipt_units(value)


def receipt_cost(value: Any) -> str | None:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        return None
    try:
        text = str(value)
        if len(text) > MAX_COST_REPRESENTATION or not text.isascii():
            return None
        amount = Decimal(text)
        if not amount.is_finite():
            return None
        digits = amount.as_tuple()
        if len(digits.digits) > MAX_COST_REPRESENTATION or not -64 <= digits.exponent <= 64:
            return None
        result = str(amount)
        return result if amount >= 0 and len(result) <= MAX_COST_REPRESENTATION else None
    except (InvalidOperation, ValueError):
        return None


def validate_unit_price(value: Any) -> float:
    if type(value) not in (int, float) or receipt_cost(value) is None:
        raise ValueError(
            "unit_price_usd must be a finite nonnegative JSON number within supported representation bounds"
        )
    return float(value)


UnitPrice = Annotated[float, BeforeValidator(validate_unit_price), Field(ge=0, allow_inf_nan=False, strict=True)]


def estimated_receipt_cost(units: Any, unit_price: float | None) -> str | None:
    count, price = receipt_units(units), receipt_cost(unit_price)
    if count is None or price is None:
        return None
    return receipt_cost(Decimal(price) * count)


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
    # Exa reports an estimate; other transports may report units without a USD price.
    cost_usd: str | None = None
    usage_units: int | None = Field(default=None, ge=0, le=2**63 - 1, strict=True)
    usage_unit: Literal["tokens", "credits"] | None = None


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
    values["cost_usd"] = receipt_cost(values.get("cost_usd"))
    units = receipt_units(values.get("usage_units"))
    unit = values.get("usage_unit")
    values["usage_units"] = units if unit in ("tokens", "credits") else None
    values["usage_unit"] = unit if units is not None and unit in ("tokens", "credits") else None
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
