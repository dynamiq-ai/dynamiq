from dynamiq import Workflow
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.connections import Pinecone
from dynamiq.nodes.embedders import OpenAITextEmbedder
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.node import InputTransformer, NodeDependency
from dynamiq.nodes.retrievers import PineconeDocumentRetriever
from dynamiq.nodes.tools.python import Python
from dynamiq.prompts import Message, Prompt
from dynamiq.storages.vector.pinecone.pinecone import PineconeIndexType
from examples.use_case_gpt_researcher.gpt_researcher.prompts import get_research_report_prompt

NUM_SOURCES_MULTIPLIER = 30


def write_report_workflow(source_to_extract: int) -> Workflow:
    """Builds a research report generation workflow."""

    # Embed the query for document retrieval
    embed_query_node = OpenAITextEmbedder(
        id="embed_query_node",
        connection=OpenAIConnection(),
        input_transformer=InputTransformer(selector={"query": "$.query"}),
    )

    # Retrieve relevant documents from Pinecone
    retrieve_documents_node = PineconeDocumentRetriever(
        id="retrieve_documents_node",
        connection=Pinecone(),
        index_name="gpt-researcher",
        index_type=PineconeIndexType.SERVERLESS,
        input_transformer=InputTransformer(
            selector={
                "embedding": f"${[embed_query_node.id]}.output.embedding",
            }
        ),
        cloud="aws",
        region="us-east-1",
        top_k=source_to_extract * NUM_SOURCES_MULTIPLIER,  # Multiplier for retrieving more sources
        depends=[NodeDependency(embed_query_node)],
    )

    curate_sources_node = Python(
        name="Python: Curate Sources",
        id="curate_sources_node",
        code="""
def run(input_data):
    documents = input_data.get("queries", {}).get("documents", [])
    max_sources = input_data.get("max_sources", float("inf"))
    max_content_per_source = input_data.get("max_content_per_source")

    grouped_content = {}

    for doc in documents:
        url = doc.metadata.get("url")
        content = doc.content

        if url:
            grouped_content.setdefault(url, []).append(content)
            if len(grouped_content) >= max_sources:
                break

    result = [{ "url": url, "content": contents[:max_content_per_source]} for url, contents in grouped_content.items()]

    return result
        """,
        input_transformer=InputTransformer(
            selector={
                "queries": f"${[retrieve_documents_node.id]}.output",
                "max_sources": "$.max_sources",
                "max_content_per_source": "$.max_content_per_source",
            }
        ),
        depends=[NodeDependency(retrieve_documents_node)],
    )

    # Generate the final research report
    generate_report_node = OpenAI(
        id="generate_report_node",
        name="Generate Research Report",
        model="gpt-4o-mini",
        connection=OpenAIConnection(),
        prompt=Prompt(messages=[Message(role="user", content=get_research_report_prompt())]),
        temperature=0.35,
        max_tokens=3000,
        input_transformer=InputTransformer(
            selector={"context": f"${[curate_sources_node.id]}.output.content", "question": "$.query"}
        ),
        depends=[NodeDependency(curate_sources_node)],
    )

    # Create workflow and add all nodes
    workflow = Workflow()
    for node in [embed_query_node, retrieve_documents_node, curate_sources_node, generate_report_node]:
        workflow.flow.add_nodes(node)

    return workflow
