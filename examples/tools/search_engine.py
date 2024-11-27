from typing import Any, Literal

from pydantic import BaseModel, Field

from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.node import ConnectionNode, Node, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger


class SearchResult(BaseModel):
    url: str
    title: str
    summary: str


class SearchEngineTool(Node):
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "Search Engine Tool"
    description: str = (
        "A tool that combines web searching, content scraping, and summarization."
    )
    search_tool: ConnectionNode = Field(
        None, description="The tool that performs the search."
    )
    scrape_tool: ConnectionNode = Field(
        None, description="The tool that scrapes the contents of the search results."
    )
    summarize_tool: Node = Field(
        None, description="The tool that summarizes the scraped contents."
    )
    default_limit: int = Field(
        5, description="The default limit for the number of search results."
    )

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {
            "search_tool": True,
            "scrape_tool": True,
            "summarize_tool": True,
        }

    def to_dict(self, **kwargs) -> dict:
        """Converts the instance to a dictionary.

        Returns:
            dict: A dictionary representation of the instance.
        """
        data = super().to_dict(**kwargs)
        data["search_tool"] = (
            self.search_tool.to_dict(**kwargs) if self.search_tool else None
        )
        data["scrape_tool"] = (
            self.scrape_tool.to_dict(**kwargs) if self.scrape_tool else None
        )
        data["summarize_tool"] = (
            self.summarize_tool.to_dict(**kwargs) if self.summarize_tool else None
        )
        return data

    def init_components(self, connection_manager: ConnectionManager | None = None):
        """
        Initialize the components of the tool.

        Args:
            connection_manager (ConnectionManager, optional): The connection manager. Defaults to ConnectionManager().
        """
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        if (
            self.search_tool is not None
            and self.search_tool.is_postponed_component_init
        ):
            self.search_tool.init_components(connection_manager)
        if (
            self.scrape_tool is not None
            and self.scrape_tool.is_postponed_component_init
        ):
            self.scrape_tool.init_components(connection_manager)
        if (
            self.summarize_tool is not None
            and self.summarize_tool.is_postponed_component_init
        ):
            self.summarize_tool.init_components(connection_manager)

    def execute(
        self, input_data: dict[str, Any], config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        query = input_data.get("input", "")
        limit = input_data.get("limit", self.default_limit)
        search_type = input_data.get("search_type", "web")
        logger.debug(
            f"Tool {self.name} - {self.id}: started with query '{query}', limit {limit}, search_type {search_type}"
        )

        try:
            search_results = self._perform_search(
                query,
                limit,
                search_type,
                config=config,
                **(kwargs | {"parent_run_id": kwargs.get("run_id")}),
            )
            scraped_contents = self._scrape_contents(
                search_results["content"]["urls"],
                config=config,
                **(kwargs | {"parent_run_id": kwargs.get("run_id")}),
            )
            summaries = self._summarize_contents(
                scraped_contents,
                config=config,
                **(kwargs | {"parent_run_id": kwargs.get("run_id")}),
            )

            formatted_results = self._format_results(
                search_results["content"]["urls"], summaries
            )

            result = {
                "search_results": search_results["content"],
                "scraped_contents": scraped_contents,
                "result": formatted_results,
                "images": search_results["content"].get("images", []),
                "sources_with_url": search_results["content"].get(
                    "sources_with_url", []
                ),
                "summaries": summaries,
            }

            logger.debug(f"Tool {self.name} - {self.id}: finished processing")
            return {"content": result}
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: execution error: {str(e)}")
            raise

    def _perform_search(
        self,
        query: str,
        limit: int,
        search_type: str,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        if self.search_tool is None:
            raise ValueError("search_tool is not initialized")
        return self.search_tool.execute(
            {"input": query, "limit": limit, "search_type": search_type},
            config=config,
            **kwargs,
        )

    def _scrape_contents(
        self,
        urls: list[str],
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        if self.scrape_tool is None:
            raise ValueError("scrape_tool is not initialized")
        scraped_contents = []
        for url in urls:
            try:
                scrape_result = self.scrape_tool.execute(
                    {"input": url},
                    config=config,
                    **kwargs,
                )
                scraped_contents.append(scrape_result["content"])
            except Exception as e:
                logger.error(f"Failed to scrape {url}: {str(e)}")
        logger.debug(
            f"Tool {self.name} - {self.id}: scraped {len(scraped_contents)} pages"
        )
        return scraped_contents

    def _summarize_contents(
        self,
        scraped_contents: list[dict[str, Any]],
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> list[str]:
        if self.summarize_tool is None:
            raise ValueError("summarize_tool is not initialized")
        summaries = []
        for content in scraped_contents:
            try:
                summary_result = self.summarize_tool.execute(
                    {"input": content["summary"]},
                    config=config,
                    **kwargs,
                )
                summaries.append(summary_result.get("content", ""))
            except Exception as e:
                logger.error(f"Failed to summarize content: {str(e)}")
        logger.debug(
            f"Tool {self.name} - {self.id}: generated {len(summaries)} summaries"
        )
        return summaries

    @staticmethod
    def _format_results(urls: list[str], summaries: list[str]) -> str:
        formatted_results = []
        for url, summary in zip(urls, summaries):
            formatted_results.extend([f"Source: {url}", f"Content: {summary}", ""])
        return "\n".join(formatted_results).strip()
