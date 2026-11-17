from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from dynamiq.connections import Context
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.nodes.types import ActionType
from dynamiq.runnables import RunnableConfig
from dynamiq.types.cancellation import check_cancellation


class ContextSearchInputSchema(BaseModel):
    """Inputs for a Context.dev web search."""

    query: str = Field(min_length=1, max_length=500, description="Public-web search query, up to 500 characters.")
    max_results: int | None = Field(default=None, ge=1, le=100, description="Maximum results, from 1 to 100.")

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Search query must not be blank.")
        return value


class _Highlights(BaseModel):
    code: str | None = None
    highlights: list[str] | None = None


class _SearchResult(BaseModel):
    url: str = Field(min_length=1)
    title: str | None = None
    description: str | None = None
    highlights: _Highlights | None = None


class _SearchResponse(BaseModel):
    results: list[_SearchResult]
    partial: bool = False


class ContextSearchTool(ConnectionNode):
    """Search the public web with Context.dev and return citable sources and passages."""

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "context-search"
    description: str = (
        "Search the public web with Context.dev for current information. "
        "Returns source URLs, titles, descriptions and available Highlights. "
        "Use a concise public query; do not include credentials or private data."
    )
    action_type: ActionType = ActionType.WEB_SEARCH
    is_parallel_execution_allowed: bool = True
    connection: Context
    max_results: int = Field(default=10, ge=1, le=100, description="Maximum number of results.")
    timeout: float = Field(default=60, gt=0, description="HTTP request timeout in seconds.")
    input_schema: ClassVar[type[ContextSearchInputSchema]] = ContextSearchInputSchema

    def _request_kwargs(self, input_data: ContextSearchInputSchema) -> dict[str, Any]:
        return {
            "method": self.connection.method,
            "url": self.connection.url,
            "headers": self.connection.headers,
            "json": {
                "query": input_data.query,
                "numResults": input_data.max_results if input_data.max_results is not None else self.max_results,
                "highlightsOptions": {"enabled": True},
            },
            "timeout": self.timeout,
        }

    def _format_response(self, response: Any) -> dict[str, Any]:
        if response.status_code >= 300:
            message = f"Context.dev returned HTTP {response.status_code}."
            guidance = {
                401: "Check your Context.dev API key.",
                403: "Check your key permissions and account access.",
                429: "Retry later and check your account limits.",
            }.get(response.status_code)
            if response.status_code >= 500:
                guidance = "Retry the search later."
            if guidance:
                message += f" {guidance}"
            raise ToolExecutionException(message, recoverable=True)
        response.raise_for_status()
        raw_response = response.json()
        try:
            data = _SearchResponse.model_validate(raw_response)
        except ValidationError:
            raise ToolExecutionException("Context.dev returned an invalid search response.", recoverable=True) from None
        sources = []
        for result in data.results:
            passages = (
                result.highlights.highlights if result.highlights and result.highlights.code == "SUCCESS" else None
            )
            sources.append(
                {
                    "url": result.url,
                    "title": result.title or result.url,
                    "content": "\n".join(passages or []) or result.description or "",
                    "highlights_status": result.highlights.code if result.highlights else "NOT_REQUESTED",
                }
            )
        text = "\n\n".join(f"[{s['title']}]({s['url']})\n{s['content']}" for s in sources)
        text = text or "No results returned by web search."
        if data.partial:
            text = "Search returned partial results.\n\n" + text
        urls = [source["url"] for source in sources]
        if self.is_optimized_for_agents:
            return {"content": text, "urls": urls, "sources": sources, "partial": data.partial}
        return {
            "content": {
                "result": text,
                "sources_with_url": [f"- [{s['title']}]({s['url']})" for s in sources],
                "urls": urls,
                "raw_response": raw_response,
                "partial": data.partial,
            }
        }

    def _wrap_exception(self, exc: Exception) -> ToolExecutionException:
        return ToolExecutionException(f"Context.dev search failed ({type(exc).__name__}).", recoverable=True)

    def execute(
        self, input_data: ContextSearchInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """Execute a search with bounded HTTP requests."""
        config = ensure_config(config)
        check_cancellation(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)
        try:
            response = self.client.request(**self._request_kwargs(input_data), allow_redirects=False)
            return self._format_response(response)
        except ToolExecutionException:
            raise
        except Exception as exc:
            raise self._wrap_exception(exc) from None

    async def execute_async(
        self, input_data: ContextSearchInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """Execute a search with the shared native async HTTP client."""
        config = ensure_config(config)
        check_cancellation(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)
        client = await self.get_async_client()
        try:
            response = await client.request(**self._request_kwargs(input_data), follow_redirects=False)
            return self._format_response(response)
        except ToolExecutionException:
            raise
        except Exception as exc:
            raise self._wrap_exception(exc) from None
