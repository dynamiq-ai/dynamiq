from dynamiq.connections import Exa
from dynamiq.nodes.tools.exa_search import (
    ContentsHighlightsOptions,
    ContentsRequest,
    ContentsSummaryOptions,
    ContentsTextOptions,
    ExaTool,
    QueryType,
)


def basic_search_example():
    """Demonstrates a minimal Exa search with date filtering."""

    exa_connection = Exa()
    exa_tool = ExaTool(connection=exa_connection, is_optimized_for_agents=False)

    result = exa_tool.run(
        input_data={
            "query": "Latest developments in quantum computing",
            "limit": 5,
            "query_type": QueryType.neural,
            "use_autoprompt": True,
            "start_crawl_date": "2024-01-01T00:00:00.000Z",
            "end_crawl_date": "2024-12-31T00:00:00.000Z",
            "moderation": False,
        }
    )

    print("Search Results:")
    print(result.output.get("content"))


def advanced_search_with_contents_example():
    """Showcases typed contents configuration, moderation, and agent formatting."""

    exa_connection = Exa()
    exa_tool = ExaTool(
        connection=exa_connection,
        is_optimized_for_agents=True,
        include_full_content=False,
    )

    result = exa_tool.run(
        input_data={
            "query": "Latest developments in quantum computing",
            "limit": 5,
            "query_type": QueryType.neural,
            "use_autoprompt": True,
            "category": "research paper",
            "start_crawl_date": "2023-01-01T00:00:00.000Z",
            "end_published_date": "2023-12-31T00:00:00.000Z",
            "include_text": ["large language model"],
            "exclude_text": ["course"],
            "context": {"maxCharacters": 8000},
            "moderation": True,
            "contents": ContentsRequest(
                text=ContentsTextOptions(max_characters=2000),
                highlights=ContentsHighlightsOptions(
                    num_sentences=2,
                    highlights_per_url=2,
                    query="Key findings",
                ),
                summary=ContentsSummaryOptions(
                    query="What breakthroughs were reported?",
                    summary_schema={
                        "$schema": "http://json-schema.org/draft-07/schema#",
                        "title": "LLM Breakthrough",
                        "type": "object",
                        "properties": {
                            "headline": {"type": "string"},
                            "impact": {"type": "string"},
                        },
                        "required": ["headline"],
                    },
                ),
                context=True,
            ),
        }
    )

    print("Search Results with Contents:")
    print(result.output.get("content"))


if __name__ == "__main__":
    basic_search_example()
    advanced_search_with_contents_example()
