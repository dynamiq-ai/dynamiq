import os

from dynamiq.connections import VertexAI
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.knowledgebases.vertex import VertexAIRagSearch
from examples.llm_setup import setup_llm

RAG_CORPUS_ID = os.getenv("VERTEXAI_RAG_CORPUS_ID", "your-rag-corpus-id")

# An LLM ranker model has to be served in the corpus's region; override if yours is elsewhere.
LLM_RANKER_MODEL = os.getenv("VERTEXAI_LLM_RANKER_MODEL", "gemini-2.5-flash")


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
    """Restrict retrieval with a metadata filter and reorder the results with the LLM ranker.

    `metadata_filter` is a CEL expression (note `==`, not `=`) over *file metadata keys*. Those
    keys are not the built-in fields such as source_display_name: they have to be declared as
    corpus metadata schemas and attached to each file, which the v1beta1 RAG data API does via
    RagDataSchema + ragMetadata:batchCreate. A filter over a key the corpus does not know matches
    nothing rather than raising, so an empty result here usually means the metadata is missing.

    The alternative ranker, `rank_service_model="semantic-ranker-default@latest"`, additionally
    needs the Discovery Engine API enabled and `discoveryengine.rankingConfigs.rank` granted to
    the calling identity; roles/aiplatform.user alone is not enough.
    """

    vertex_connection = VertexAI()
    rag_tool = VertexAIRagSearch(
        connection=vertex_connection,
        rag_corpus_id=RAG_CORPUS_ID,
        top_k=10,
        vector_distance_threshold=0.6,
        llm_ranker_model=LLM_RANKER_MODEL,
    )

    result = rag_tool.run(
        input_data={
            "query": "What is our refund policy?",
            "metadata_filter": 'department == "support" && year == 2025',
        }
    )

    print("Filtered content:")
    print(result.output.get("content"))
    for document in result.output.get("documents", []):
        # A ranker changes the order, not the score: `score` stays the vector score and
        # `reranked` records that the ordering came from the ranker.
        print(document.metadata.get("source_uri"), document.score, document.metadata.get("reranked"))


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
