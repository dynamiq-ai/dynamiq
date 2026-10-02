from dynamiq.connections import Linkup
from dynamiq.nodes.tools.linkup_search import LinkupTool


def basic_search_example():
    """Demonstrates a minimal Linkup search with domain and date filtering."""

    linkup_connection = Linkup()
    linkup_tool = LinkupTool(connection=linkup_connection, is_optimized_for_agents=False)

    result = linkup_tool.run(
        input_data={
            "query": "Latest developments in quantum computing",
            "max_results": 5,
            "exclude_domains": ["wikipedia.org"],
            "from_date": "2026-01-01",
        }
    )

    print("Search Results:")
    print(result.output.get("content"))


def sourced_answer_example():
    """Showcases a deep search returning a sourced answer in agent-optimized format."""

    linkup_connection = Linkup()
    linkup_tool = LinkupTool(
        connection=linkup_connection,
        is_optimized_for_agents=True,
        depth="deep",
        output_type="sourcedAnswer",
    )

    result = linkup_tool.run(
        input_data={
            "query": "What breakthroughs in quantum error correction were reported this year?",
        }
    )

    print("Sourced Answer:")
    print(result.output.get("content"))


if __name__ == "__main__":
    basic_search_example()
    sourced_answer_example()
