"""
Example of using Elasticsearch as a vector store in a RAG workflow.
This example demonstrates:
1. Document conversion and splitting
2. Vector embedding generation
3. Storage in Elasticsearch
4. Vector similarity search
5. Answer generation using retrieved context
"""

from io import BytesIO

from dynamiq import Workflow
from dynamiq.connections import Elasticsearch as ElasticsearchConnection
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.converters import PyPDFConverter
from dynamiq.nodes.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.retrievers import ElasticsearchDocumentRetriever
from dynamiq.nodes.splitters.document import DocumentSplitter
from dynamiq.nodes.writers import ElasticsearchDocumentWriter
from dynamiq.prompts import Message, Prompt

# Initialize Elasticsearch connection
es_connection = ElasticsearchConnection()

# Initialize OpenAI connection
openai_connection = OpenAIConnection()


def create_indexing_workflow(files: list[str]) -> None:
    """Create and run the document indexing workflow."""

    # Initialize workflow
    rag_wf = Workflow()

    # PDF converter
    converter = PyPDFConverter(document_creation_mode="one-doc-per-page")
    rag_wf.flow.add_nodes(converter)

    # Document splitter
    document_splitter = (
        DocumentSplitter(
            split_by="sentence",
            split_length=10,
            split_overlap=1,
        )
        .inputs(documents=converter.outputs.documents)
        .depends_on(converter)
    )
    rag_wf.flow.add_nodes(document_splitter)

    # OpenAI embeddings
    embedder = (
        OpenAIDocumentEmbedder(
            connection=openai_connection,
            model="text-embedding-3-small",
        )
        .inputs(documents=document_splitter.outputs.documents)
        .depends_on(document_splitter)
    )
    rag_wf.flow.add_nodes(embedder)

    # Elasticsearch vector store
    vector_store = (
        ElasticsearchDocumentWriter(
            connection=es_connection,
            index_name="documents",
            dimension=1536,
            similarity="cosine",
            create_if_not_exist=True,
            index_settings={
                "number_of_shards": 1,
                "number_of_replicas": 1,
            },
            mapping_settings={
                "dynamic": "strict",
            },
        )
        .inputs(documents=embedder.outputs.documents)
        .depends_on(embedder)
    )
    rag_wf.flow.add_nodes(vector_store)

    # Prepare input data
    input_data = {
        "files": [BytesIO(open(path, "rb").read()) for path in files],
        "metadata": [{"filename": path} for path in files],
    }

    # Run indexing workflow
    rag_wf.run(input_data=input_data)
    print(f"Successfully indexed {len(files)} documents")


def create_retrieval_workflow(question: str) -> str:
    """Create and run the document retrieval workflow."""

    # Initialize workflow
    retrieval_wf = Workflow()

    # Query embedder
    embedder = OpenAITextEmbedder(
        connection=openai_connection,
        model="text-embedding-3-small",
    )
    retrieval_wf.flow.add_nodes(embedder)

    # Elasticsearch retriever
    document_retriever = (
        ElasticsearchDocumentRetriever(
            connection=es_connection,
            index_name="documents",
            top_k=5,
        )
        .inputs(
            query=embedder.outputs.embedding,  # Pass embedding for vector search
            scale_scores=True,  # Scale scores to 0-1 range
        )
        .depends_on(embedder)
    )
    retrieval_wf.flow.add_nodes(document_retriever)

    # Answer generation
    prompt_template = """
    Please answer the question based on the provided context.

    Question: {{ query }}

    Context:
    {% for document in documents %}
    - {{ document.content }} (Relevance Score: {{ document.score }})
    {% endfor %}

    Answer:
    """

    prompt = Prompt(messages=[Message(content=prompt_template, role="user")])

    answer_generator = (
        OpenAI(
            connection=openai_connection,
            model="gpt-4",
            prompt=prompt,
        )
        .inputs(
            documents=document_retriever.outputs.documents,
            query=embedder.outputs.query,
        )
        .depends_on([document_retriever, embedder])
    )
    retrieval_wf.flow.add_nodes(answer_generator)

    # Run retrieval workflow
    result = retrieval_wf.run(
        input_data={
            "query": question,
        }
    )
    return result.output.get(answer_generator.id).get("output", {}).get("content")


def main():
    # Example usage
    files = ["example_file.pdf"]

    # Index documents
    create_indexing_workflow(files)

    # Ask questions
    questions = [
        "Name shown on tax return?",
        "Current name, address?",
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        answer = create_retrieval_workflow(question)
        print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
