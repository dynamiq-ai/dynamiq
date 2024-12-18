from dynamiq.connections import Jina
from dynamiq.nodes.tools import JinaResponseFormat, JinaScrapeTool, JinaSearchTool


def basic_search_example():
    jina_connection = Jina()

    jina_search_tool = JinaSearchTool(connection=jina_connection, is_optimized_for_agents=False)

    result = jina_search_tool.run(
        input_data={
            "query": "What is LLM?",
            "max_results": 3,
        }
    )

    print("Search Results:")
    print(result.output.get("content"))


def basic_scrape_example():
    jina_connection = Jina()

    jina_search_tool = JinaScrapeTool(
        connection=jina_connection, is_optimized_for_agents=False, response_format=JinaResponseFormat.DEFAULT
    )

    result = jina_search_tool.run(
        input_data={
            "url": "https://example.com",
        }
    )

    print("Scrape Results:")
    print(result.output.get("content"))


if __name__ == "__main__":
    basic_search_example()
    basic_scrape_example()
