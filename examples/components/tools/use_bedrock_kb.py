import os

from dynamiq.connections import AWS
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.knowledgebases.bedrock import BedrockKnowledgeBaseSearch
from examples.llm_setup import setup_llm

KNOWLEDGE_BASE_ID = os.getenv("BEDROCK_KNOWLEDGE_BASE_ID", "your-knowledge-base-id")


def basic_search_example():
    """Retrieve chunks from a Bedrock Knowledge Base with default settings."""

    aws_connection = AWS()
    kb_tool = BedrockKnowledgeBaseSearch(
        connection=aws_connection,
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        top_k=5,
    )

    result = kb_tool.run(input_data={"query": "What is our refund policy?"})

    print("Retrieved content:")
    print(result.output.get("content"))


def filtered_search_example():
    """Restrict retrieval with a Bedrock RetrievalFilter and a relevance score threshold."""

    aws_connection = AWS()
    kb_tool = BedrockKnowledgeBaseSearch(
        connection=aws_connection,
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        top_k=10,
        search_type="SEMANTIC",
        score_threshold=0.4,
    )

    result = kb_tool.run(
        input_data={
            "query": "What is our refund policy?",
            "filters": {
                "andAll": [
                    {"equals": {"key": "department", "value": "support"}},
                    {"greaterThan": {"key": "year", "value": 2023}},
                ]
            },
        }
    )

    print("Filtered content:")
    print(result.output.get("content"))
    for document in result.output.get("documents", []):
        print(document.metadata.get("source_uri"), document.score)


def agent_with_bedrock_kb_example():
    """Give an agent native access to a Bedrock Knowledge Base as a retrieval tool."""

    llm = setup_llm()
    kb_tool = BedrockKnowledgeBaseSearch(
        connection=AWS(),
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        top_k=5,
    )

    agent = Agent(
        name="support-agent",
        llm=llm,
        tools=[kb_tool],
        role="A support assistant that answers questions strictly from the company knowledge base.",
    )

    result = agent.run(input_data={"input": "Summarize our refund policy and cite the source documents."})
    print(result.output.get("content"))


if __name__ == "__main__":
    basic_search_example()
    filtered_search_example()
    agent_with_bedrock_kb_example()
