from dynamiq.connections import Exa
from dynamiq.nodes.tools.exa_search import ExaInputSchema, ExaTool, QueryType


def basic_search_example():
    exa_connection = Exa()

    exa_tool = ExaTool(connection=exa_connection, is_optimized_for_agents=False)

    result = exa_tool.run(
        input_data={
            "query": "Latest developments in quantum computing",
            "limit": 5,
            "query_type": QueryType.neural,
            "use_autoprompt": True,
        }
    )

    print("Search Results:")
    print(result.output.get("content"))


def advanced_search_with_contents_example():
    exa_connection = Exa()
    exa_tool = ExaTool(connection=exa_connection, is_optimized_for_agents=True)

    result = exa_tool.run(
        input_data=ExaInputSchema(
            query="AI breakthroughs in healthcare",
            limit=5,
            query_type=QueryType.neural,
            use_autoprompt=True,
            category="research paper",
            include_full_content=True,
        )
    )

    print("Search Results with Contents:")
    print(result.output.get("content"))


if __name__ == "__main__":
    basic_search_example()
    advanced_search_with_contents_example()
