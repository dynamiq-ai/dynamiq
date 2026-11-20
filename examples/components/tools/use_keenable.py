from dynamiq.connections import Keenable
from dynamiq.nodes.tools.keenable_search import KeenableTool


def basic_search_example():
    """Keenable works without an API key; set KEENABLE_API_KEY for higher limits."""

    tool = KeenableTool(connection=Keenable(), is_optimized_for_agents=False)
    result = tool.run(input_data={"query": "Latest developments in quantum computing", "max_results": 5})

    print("Search Results:")
    print(result.output.get("content"))


if __name__ == "__main__":
    basic_search_example()
