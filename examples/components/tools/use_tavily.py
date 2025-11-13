from dynamiq.connections import Tavily
from dynamiq.nodes.tools.tavily import TavilyTool


def basic_search_example():
    """Example of basic Tavily usage with minimal overrides."""
    tavily_connection = Tavily()

    # Initialize the tool with default settings
    tavily_tool = TavilyTool(connection=tavily_connection, is_optimized_for_agents=False)

    # Run a basic search
    result = tavily_tool.run(
        input_data={
            "query": "Latest developments in quantum computing",
            "search_depth": "basic",
            "max_results": 5,
            "include_answer": True,
            "use_cache": True,
            "auto_parameters": False,
        }
    )

    print("=== BASIC SEARCH RESULTS ===")
    print(result.output.get("content"))


def optimized_advanced_search_example():
    """Demonstrates the agent-optimized view with advanced depth and media outputs."""
    tavily_connection = Tavily()

    tavily_tool = TavilyTool(
        connection=tavily_connection,
        is_optimized_for_agents=True,
        search_depth="advanced",
        include_answer="advanced",
        include_images=True,
        include_image_descriptions=True,
        include_favicon=True,
        include_raw_content="markdown",
        max_results=4,
    )

    result = tavily_tool.run(
        input_data={
            "query": "Key breakthroughs in energy-efficient data centers",
            "topic": "news",
            "time_range": "month",
            "chunks_per_source": 3,
            "include_domains": ["datacenterdynamics.com", "techcrunch.com"],
            "country": "united states",
            "auto_parameters": False,
        }
    )

    print("=== ADVANCED AGENT-READY RESULTS ===")
    print(result.output.get("content"))


if __name__ == "__main__":
    basic_search_example()
    optimized_advanced_search_example()
