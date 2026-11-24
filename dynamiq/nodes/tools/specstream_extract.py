from typing import Any, ClassVar, Literal
from urllib.parse import urljoin

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.connections import SpecStream
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.nodes.tools.utils import http_error_detail
from dynamiq.nodes.types import ActionType
from dynamiq.runnables import RunnableConfig
from dynamiq.types.cancellation import check_cancellation
from dynamiq.utils.logger import logger

DESCRIPTION_SPECSTREAM_EXTRACT = """Extracts clean text or markdown from one or more web URLs via SpecStream.

Key capabilities:
- Batch extraction for up to 20 URLs per request
- Returns clean title and page text/markdown with source attribution

Examples:
- {"url": "https://example.com/article"}
- {"urls": ["https://example.com/a", "https://example.com/b"]}
"""


class SpecStreamExtractInputSchema(BaseModel):
    """Schema for SpecStream page extraction input parameters."""

    url: str | None = Field(default=None, description="Single URL to extract.")
    urls: list[str] | None = Field(default=None, description="List of URLs to extract (up to 20).")
    brief: str = Field(
        default="Extracting webpage content via SpecStream.",
        description="Very brief description of the extraction action.",
    )


class SpecStreamExtractTool(ConnectionNode):
    """A tool for extracting clean webpage content using the SpecStream Extract API."""

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "specstream-extract"
    description: str = DESCRIPTION_SPECSTREAM_EXTRACT
    action_type: ActionType = ActionType.WEB_SCRAPE
    is_parallel_execution_allowed: bool = True
    connection: SpecStream
    url: str | None = Field(default=None, description="Optional default URL to extract.")

    model_config = ConfigDict(arbitrary_types_allowed=True)
    input_schema: ClassVar[type[SpecStreamExtractInputSchema]] = SpecStreamExtractInputSchema

    def _resolve_extract_request(self, input_data: SpecStreamExtractInputSchema) -> tuple[list[str], dict[str, Any], str]:
        urls: list[str] = []
        if input_data.urls:
            urls.extend(u.strip() for u in input_data.urls if u and u.strip())
        single = input_data.url or self.url
        if single and single.strip() and single.strip() not in urls:
            urls.append(single.strip())
        if not urls:
            raise ValueError("At least one URL is required for extraction")
        payload = {"urls": urls[:20]}
        connection_url = urljoin(self.connection.url.rstrip("/") + "/", "extract")
        return urls[:20], payload, connection_url

    def _wrap_extract_exception(self, exc: Exception) -> ToolExecutionException:
        error = http_error_detail(exc)
        logger.error(f"Tool {self.name} - {self.id}: failed to extract URLs. Error: {error}")
        return ToolExecutionException(
            f"Tool '{self.name}' failed to extract webpage content. Error: {error}. "
            f"Please analyze the error and take appropriate action.",
            recoverable=True,
        )

    def _format_extract_output(self, urls: list[str], extract_result: dict[str, Any]) -> dict[str, Any]:
        raw_results = [r for r in (extract_result.get("results") or []) if isinstance(r, dict)]
        if self.is_optimized_for_agents:
            sections = []
            for item in raw_results:
                item_url = item.get("url") or ""
                title = item.get("title") or item_url
                text = item.get("markdown") or item.get("text") or ""
                if item.get("ok", True) and text:
                    sections.append(f"## {title}\nURL: {item_url}\n\n{text}")
                else:
                    err = item.get("error") or "No extractable content"
                    sections.append(f"## {title}\nURL: {item_url}\n\nExtraction failed: {err}")
            return {"content": "\n\n---\n\n".join(sections), "urls": urls, "results": raw_results}

        return {"content": {"urls": urls, "results": raw_results, "raw_response": extract_result}}

    def execute(
        self, input_data: SpecStreamExtractInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        config = ensure_config(config)
        check_cancellation(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        urls, payload, connection_url = self._resolve_extract_request(input_data)
        try:
            response = self.client.request(
                method=self.connection.method,
                url=connection_url,
                json=payload,
                headers=self.connection.headers,
            )
            response.raise_for_status()
            extract_result = response.json()
        except Exception as e:
            raise self._wrap_extract_exception(e) from e

        return self._format_extract_output(urls, extract_result)

    async def execute_async(
        self, input_data: SpecStreamExtractInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        config = ensure_config(config)
        check_cancellation(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        urls, payload, connection_url = self._resolve_extract_request(input_data)
        client = await self.get_async_client()
        try:
            response = await client.request(
                method=self.connection.method,
                url=connection_url,
                json=payload,
                headers=self.connection.headers,
            )
            response.raise_for_status()
            extract_result = response.json()
        except Exception as e:
            raise self._wrap_extract_exception(e) from e

        return self._format_extract_output(urls, extract_result)
