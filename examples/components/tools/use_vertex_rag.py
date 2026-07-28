import os

from dynamiq.connections import VertexAI
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.knowledgebases.vertex import VertexAIRagSearch
from examples.llm_setup import setup_llm

RAG_CORPUS_ID = os.getenv("VERTEXAI_RAG_CORPUS_ID", "your-rag-corpus-id")


def basic_search_example():
    """Retrieve chunks from a Vertex AI RAG Engine corpus with default settings."""

    vertex_connection = VertexAI()
    rag_tool = VertexAIRagSearch(
        connection=vertex_connection,
        rag_corpus_id=RAG_CORPUS_ID,
        top_k=5,
    )

    result = rag_tool.run(input_data={"query": "What is our refund policy?"})

    print("Retrieved content:")
    print(result.output.get("content"))


def filtered_and_ranked_search_example():
    """Restrict retrieval with a metadata filter and rerank results with the semantic ranker."""

    vertex_connection = VertexAI()
    rag_tool = VertexAIRagSearch(
        connection=vertex_connection,
        rag_corpus_id=RAG_CORPUS_ID,
        top_k=10,
        vector_distance_threshold=0.6,
        rank_service_model="semantic-ranker-default@latest",
    )

    result = rag_tool.run(
        input_data={
            "query": "What is our refund policy?",
            "metadata_filter": 'source_display_name = "policies.pdf"',
        }
    )

    print("Filtered content:")
    print(result.output.get("content"))
    for document in result.output.get("documents", []):
        print(document.metadata.get("source_uri"), document.score)


def agent_with_vertex_rag_example():
    """Give an agent native access to a Vertex AI RAG Engine corpus as a retrieval tool."""

    llm = setup_llm()
    rag_tool = VertexAIRagSearch(
        connection=VertexAI(),
        rag_corpus_id=RAG_CORPUS_ID,
        top_k=5,
    )

    agent = Agent(
        name="support-agent",
        llm=llm,
        tools=[rag_tool],
        role="A support assistant that answers questions strictly from the company knowledge base.",
    )

    result = agent.run(input_data={"input": "Summarize our refund policy and cite the source documents."})
    print(result.output.get("content"))


if __name__ == "__main__":
    basic_search_example()
    filtered_and_ranked_search_example()
    agent_with_vertex_rag_example()
