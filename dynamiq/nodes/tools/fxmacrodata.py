from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.connections import FXMacroData
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.types.cancellation import check_cancellation
from dynamiq.utils.logger import logger

FXMacroDataEndpoint = Literal["announcements", "calendar", "data_catalogue", "forex"]

DESCRIPTION_FXMACRODATA = """Retrieves official macroeconomic releases, release calendars and FX rates from FXMacroData.

Key Capabilities:
- Indicator history with release timestamps and source links (CPI, GDP, policy rates, payrolls, bond yields)
- Upcoming release calendar per currency, with times in UTC and the publisher's local time
- Data catalogue listing the indicator slugs available for a currency
- Daily FX rates for currency pairs

Usage Strategy:
Call "data_catalogue" first when you do not know the indicator slug for a currency.
Use "announcements" for released values, "calendar" for what is scheduled and "forex" for exchange rates.
Currencies are ISO codes such as usd, eur, gbp, jpy, aud. Dates use YYYY-MM-DD.
USD data and the catalogue work without an API key; other currencies and FX rates need one.

Examples:
- Latest US CPI: {"endpoint": "announcements", "currency": "usd", "indicator": "inflation", "limit": 1}
- ECB decisions this year: {"endpoint": "announcements", "currency": "eur", "indicator": "policy_rate", "start_date": "2026-01-01"}
- Upcoming US releases: {"endpoint": "calendar", "currency": "usd"}
- Available Japanese indicators: {"endpoint": "data_catalogue", "currency": "jpy"}
- EUR/USD history: {"endpoint": "forex", "currency": "eur", "quote": "usd", "start_date": "2026-09-01"}"""  # noqa: E501


class FXMacroDataInputSchema(BaseModel):
    endpoint: FXMacroDataEndpoint = Field(
        default="announcements",
        description="Which data to fetch: 'announcements', 'calendar', 'data_catalogue' or 'forex'.",
    )
    currency: str = Field(
        default="",
        description="ISO currency code such as 'usd' or 'eur'. For 'forex' this is the base currency.",
    )
    indicator: str | None = Field(
        default=None,
        description="Indicator slug such as 'inflation' or 'policy_rate'. Required for 'announcements', "
        "optional filter for 'calendar'.",
    )
    quote: str | None = Field(default=None, description="Quote currency for 'forex', for example 'usd'.")
    start_date: str | None = Field(default=None, description="Start date (YYYY-MM-DD).")
    end_date: str | None = Field(default=None, description="End date (YYYY-MM-DD).")
    limit: int | None = Field(
        default=None, ge=1, le=100, description="Number of rows to return for 'announcements' and 'forex' (1-100)."
    )


class FXMacroDataTool(ConnectionNode):
    """
    A tool for retrieving macroeconomic and FX data from the FXMacroData API.

    One tool covers four read-only endpoints: indicator announcements, the release calendar,
    the data catalogue and FX rates. Responses keep the API's own field names, units and
    source links; the agent-optimized output adds a compact text rendering on top.

    Attributes:
        group (Literal[NodeGroup.TOOLS]): The group to which this tool belongs.
        name (str): The name of the tool.
        description (str): A brief description of the tool.
        connection (FXMacroData): The connection instance for the FXMacroData API.
        endpoint (str): The default endpoint to call.
        currency (str): The default currency code.
        indicator (str): The default indicator slug.
        limit (int): The default number of rows for announcements and FX rates.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "fxmacrodata"
    description: str = DESCRIPTION_FXMACRODATA
    is_parallel_execution_allowed: bool = True
    connection: FXMacroData

    endpoint: FXMacroDataEndpoint = Field(default="announcements", description="The default endpoint to call")
    currency: str = Field(default="", description="The default ISO currency code")
    indicator: str = Field(default="", description="The default indicator slug")
    limit: int = Field(default=20, ge=1, le=100, description="The default number of rows to return")
    timeout: float = Field(default=30.0, gt=0, description="Request timeout in seconds")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_schema: ClassVar[type[FXMacroDataInputSchema]] = FXMacroDataInputSchema

    @staticmethod
    def _code(value: str | None, field: str) -> str:
        code = (value or "").strip().lower()
        if len(code) != 3 or not code.isalpha():
            raise ToolExecutionException(
                f"Parameter '{field}' must be a three-letter currency code such as 'usd', got '{value}'.",
                recoverable=True,
            )
        return code

    def _build_request_kwargs(self, input_data: FXMacroDataInputSchema) -> tuple[dict[str, Any], str]:
        """Return (request_kwargs, endpoint)."""
        if self.connection.api_key and not self.connection.has_valid_api_key:
            raise ToolExecutionException(
                "The FXMacroData API key contains whitespace or non-printable characters. "
                "Check the FXMACRODATA_API_KEY value.",
                recoverable=False,
            )
        endpoint = input_data.endpoint or self.endpoint
        currency = self._code(input_data.currency or self.currency, "currency")
        indicator = (input_data.indicator or self.indicator or "").strip().lower()
        limit = input_data.limit or self.limit

        params: dict[str, Any] = {}
        if endpoint == "announcements":
            if not indicator:
                raise ToolExecutionException(
                    "Parameter 'indicator' is required for the 'announcements' endpoint. "
                    "Call the 'data_catalogue' endpoint to list the slugs for a currency.",
                    recoverable=True,
                )
            path = f"/v1/announcements/{currency}/{indicator}"
            params["limit"] = limit
        elif endpoint == "calendar":
            path = f"/v1/calendar/{currency}"
            if indicator:
                params["indicator"] = indicator
        elif endpoint == "data_catalogue":
            path = f"/v1/data_catalogue/{currency}"
        else:
            quote = self._code(input_data.quote, "quote")
            path = f"/v1/forex/{currency}/{quote}"
            params["limit"] = limit

        if endpoint != "data_catalogue":
            if input_data.start_date:
                params["start_date"] = input_data.start_date
            if input_data.end_date:
                params["end_date"] = input_data.end_date

        request_kwargs = {
            "method": self.connection.method,
            "url": f"{self.connection.url}{path}",
            "headers": self.connection.headers,
            "params": params,
            "timeout": self.timeout,
        }
        return request_kwargs, endpoint

    @staticmethod
    def _format_rows(endpoint: str, payload: Any) -> tuple[str, list[str]]:
        """Render the response as compact text and collect source links."""
        lines: list[str] = []
        sources: list[str] = []

        if endpoint == "data_catalogue":
            entries = payload.items() if isinstance(payload, dict) else []
            for slug, meta in entries:
                meta = meta if isinstance(meta, dict) else {}
                details = ", ".join(str(meta[k]) for k in ("unit", "source") if meta.get(k))
                lines.append(f"{slug}: {meta.get('name', slug)}" + (f" ({details})" if details else ""))
            return "\n".join(lines), sources

        rows = payload.get("data", []) if isinstance(payload, dict) else []
        if endpoint == "announcements":
            name = payload.get("name") or payload.get("indicator")
            lines.append(f"{name} ({payload.get('currency')}), source: {payload.get('source')}")
            for row in rows:
                released = row.get("announcement_datetime_local") or "release time unknown"
                lines.append(f"{row.get('date')}: {row.get('val')} (released {released})")
                if row.get("source_url"):
                    sources.append(f"[{name} {row.get('date')}]({row.get('source_url')})")
        elif endpoint == "calendar":
            for row in rows:
                when = row.get("announcement_datetime_local") or row.get("announcement_datetime_utc")
                line = f"{when}: {row.get('name')} ({row.get('release')})"
                if row.get("event_importance"):
                    line += f", importance {row.get('event_importance')}"
                lines.append(line)
        else:
            lines.append(f"{payload.get('base')}/{payload.get('quote')}, source: {payload.get('source')}")
            for row in rows:
                lines.append(f"{row.get('date')}: {row.get('val')}")

        return "\n".join(lines), sources

    def _redact(self, text: str) -> str:
        """Remove the API key from text that ends up in logs or tool output."""
        key = self.connection.api_key
        return text.replace(key, "[redacted]") if key else text

    def _fail(self, error: str) -> ToolExecutionException:
        error = self._redact(error)
        logger.error(f"Tool {self.name} - {self.id}: failed to get data. Error: {error}")
        return ToolExecutionException(
            f"Tool '{self.name}' failed to retrieve data. "
            f"Error: {error}. Please analyze the error and take appropriate action.",
            recoverable=True,
        )

    def _validate_payload(self, endpoint: str, payload: Any) -> None:
        """Reject error bodies served with HTTP 200 and responses of the wrong shape."""
        if isinstance(payload, dict) and ("detail" in payload or "error" in payload) and "data" not in payload:
            raise self._fail(f"API error: {payload.get('detail') or payload.get('error')}")
        if endpoint == "data_catalogue":
            if not isinstance(payload, dict):
                raise self._fail("Unexpected response shape: expected a JSON object")
            return
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise self._fail("Unexpected response shape: expected an object with a 'data' list")
        if not all(isinstance(row, dict) for row in payload["data"]):
            raise self._fail("Unexpected response shape: 'data' rows must be objects")

    def _handle_response(self, response: Any, endpoint: str) -> dict[str, Any]:
        if 300 <= response.status_code < 400:
            # Redirects are not followed, so the key header is never sent to another host.
            raise self._fail(f"HTTP {response.status_code}: redirect refused")
        if response.status_code >= 400:
            # Error bodies are {"detail": ...}; 401 also carries the subscription link.
            try:
                body = response.json()
                error = body.get("detail", body) if isinstance(body, dict) else body
            except ValueError:
                error = response.text
            raise self._fail(f"HTTP {response.status_code}: {error}")

        try:
            payload = response.json()
        except ValueError:
            raise self._fail("The API returned a response that is not JSON") from None
        self._validate_payload(endpoint, payload)
        formatted, sources = self._format_rows(endpoint, payload)

        if self.is_optimized_for_agents:
            content = f"## FXMacroData {endpoint}\n{formatted}"
            if sources:
                content = "## Sources with URLs\n" + "\n".join(sources) + "\n\n" + content
            return {"content": content}

        return {
            "content": {
                "result": formatted,
                "sources_with_url": sources,
                "raw_response": payload,
            }
        }

    def _wrap_request_exception(self, exc: Exception) -> ToolExecutionException:
        error = self._redact(str(exc))
        logger.error(f"Tool {self.name} - {self.id}: unexpected error occurred. Error: {error}")
        return ToolExecutionException(
            f"Tool '{self.name}' encountered an unexpected error. "
            f"Error: {error}. Please analyze the error and take appropriate action.",
            recoverable=True,
        )

    def execute(
        self, input_data: FXMacroDataInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Calls the selected FXMacroData endpoint and returns the formatted result.
        """
        config = ensure_config(config)
        check_cancellation(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        request_kwargs, endpoint = self._build_request_kwargs(input_data)
        try:
            response = self.client.request(**request_kwargs, allow_redirects=False)
            return self._handle_response(response, endpoint)
        except ToolExecutionException:
            raise
        except Exception as e:
            raise self._wrap_request_exception(e)

    async def execute_async(
        self, input_data: FXMacroDataInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """Native async execution path mirroring ``execute``."""
        config = ensure_config(config)
        check_cancellation(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        request_kwargs, endpoint = self._build_request_kwargs(input_data)
        client = await self.get_async_client()
        try:
            response = await client.request(**request_kwargs, follow_redirects=False)
            return self._handle_response(response, endpoint)
        except ToolExecutionException:
            raise
        except Exception as e:
            raise self._wrap_request_exception(e)
