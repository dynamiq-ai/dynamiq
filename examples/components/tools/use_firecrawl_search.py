from dynamiq.connections import Firecrawl
from dynamiq.nodes.tools.firecrawl_search import FirecrawlSearchTool, SourceNews, SourceWeb


def basic_search_example():
    """Run a default Firecrawl search with web results only."""
    firecrawl_connection = Firecrawl()

    search_tool = FirecrawlSearchTool(connection=firecrawl_connection, is_optimized_for_agents=False)

    result = search_tool.run(
        input_data={
            "query": "firecrawl search API",
            "limit": 5,
        }
    )

    print("=== BASIC SEARCH RESULTS ===")
    print(result.output.get("content"))


def search_with_scrape_example():
    """Search across web/news with category scoping."""
    firecrawl_connection = Firecrawl()

    search_tool = FirecrawlSearchTool(
        connection=firecrawl_connection,
        is_optimized_for_agents=True,
        limit=3,
        sources=[SourceWeb(tbs="qdr:w"), SourceNews()],
        categories=["github"],
        country="US",
    )

    result = search_tool.run(
        input_data={
            "query": "firecrawl github examples",
            "ignoreInvalidURLs": True,
            "timeout": 45000,
        }
    )

    print("=== SEARCH + SCRAPE (AGENT FORMAT) ===")
    print(result.output.get("content"))


if __name__ == "__main__":
    basic_search_example()
    search_with_scrape_example()
