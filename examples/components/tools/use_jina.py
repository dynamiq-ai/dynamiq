from dynamiq.connections import Jina
from dynamiq.nodes.tools import JinaResponseFormat, JinaScrapeTool, JinaSearchTool


def agent_ready_search_example():
    """Demonstrate an agent-optimized search that concatenates every SERP entry."""

    jina_connection = Jina()

    jina_search_tool = JinaSearchTool(
        connection=jina_connection,
        is_optimized_for_agents=True,
        include_full_content=True,
        include_links=True,
        include_images="all",
        include_favicons=True,
    )

    result = jina_search_tool.run(
        input_data={
            "query": "Latest techniques to cut LLM inference cost",
            "site": "https://jina.ai",
            "country": "US",
            "language": "en",
            "max_results": 5,
            "return_format": JinaResponseFormat.MARKDOWN,
            "no_cache": True,
            "timeout": 20,
        }
    )

    print("=== Agent-Optimized Search Result ===")
    print(result.output.get("content"))


def structured_search_example():
    """Demonstrate structured (non-agent) output with advanced headers."""

    jina_connection = Jina()

    jina_search_tool = JinaSearchTool(
        connection=jina_connection,
        is_optimized_for_agents=False,
        include_full_content=False,
        include_links="all",
        include_images=False,
        respond_with="no-content",
        engine="direct",
    )

    result = jina_search_tool.run(
        input_data={
            "query": "Composable agent frameworks overview",
            "max_results": 3,
            "locale": "en-US",
            "retain_images": "none",
            "include_favicon": True,
        }
    )

    print("=== Structured Search Payload ===")
    print(result.output.get("content"))


def advanced_scrape_example():
    """Demonstrate reader usage that mirrors common scraper tweaks."""

    jina_connection = Jina()

    jina_scrape_tool = JinaScrapeTool(
        connection=jina_connection,
        is_optimized_for_agents=False,
        response_format=JinaResponseFormat.MARKDOWN,
        include_links=True,
        include_images=True,
        engine="browser",
    )

    result = jina_scrape_tool.run(
        input_data={
            "url": "https://jina.ai/news",
            "target_selector": "main",
            "remove_selector": "nav,footer,.ads",
            "no_cache": True,
        }
    )

    print("=== Reader Result (Structured) ===")
    print(result.output.get("content"))


if __name__ == "__main__":
    agent_ready_search_example()
    structured_search_example()
    advanced_scrape_example()
