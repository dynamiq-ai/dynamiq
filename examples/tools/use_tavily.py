from dynamiq.connections import Tavily
from dynamiq.nodes.tools.tavily import TavilyTool


def basic_search_example():
    """Example of basic search using TavilyTool."""
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
        }
    )

    print("Basic Search Results:")
    print(result.output.get("content"))


def search_with_parameter_override():
    """Example demonstrating parameter override during execution."""
    tavily_connection = Tavily()

    tavily_tool = TavilyTool(connection=tavily_connection, search_depth="basic", max_results=5, include_answer=False)

    result = tavily_tool.run(
        input_data={
            "query": "Latest developments in quantum computing",
            "search_depth": "advanced",
            "max_results": 3,
            "include_answer": True,
            "exclude_domains": ["wikipedia.org"],
        }
    )

    print("Search Results with Parameter Override:")
    print(result.output.get("content"))


if __name__ == "__main__":
    basic_search_example()
    search_with_parameter_override()
