from dynamiq.connections import ScaleSerp
from dynamiq.nodes.tools.scale_serp import ScaleSerpTool, SearchType


def basic_search_example():
    scale_connection = ScaleSerp()

    # Initialize with default parameters
    search_tool = ScaleSerpTool(connection=scale_connection, is_optimized_for_agents=False, limit=5)

    result = search_tool.run(
        input_data={"query": "Latest developments in artificial intelligence", "search_type": SearchType.WEB}
    )

    print("Basic Search Results:")
    print(result.output.get("content"))


def advanced_search_example():
    scale_connection = ScaleSerp()

    # Initialize with specific parameters
    search_tool = ScaleSerpTool(
        connection=scale_connection, is_optimized_for_agents=True, search_type=SearchType.NEWS, limit=10
    )

    # Override some parameters during execution
    result = search_tool.run(
        input_data={"query": "Latest developments in artificial intelligence", "limit": 5}  # Override the default limit
    )

    print("Advanced Search Results:")
    print(result.output.get("content"))


if __name__ == "__main__":
    basic_search_example()
    advanced_search_example()
