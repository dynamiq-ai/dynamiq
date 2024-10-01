from dynamiq import Workflow
from dynamiq.connections.managers import ConnectionManager
from dynamiq.flows import Flow
from dynamiq.nodes import InputTransformer
from dynamiq.nodes.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from dynamiq.nodes.node import NodeDependency
from dynamiq.nodes.retrievers import (
    ChromaDocumentRetriever,
    PineconeDocumentRetriever,
    QdrantDocumentRetriever,
    WeaviateDocumentRetriever,
)
from dynamiq.nodes.writers import (
    ChromaDocumentWriter,
    PineconeDocumentWriter,
    QdrantDocumentWriter,
    WeaviateDocumentWriter,
)
from dynamiq.storages.vector import ChromaVectorStore, PineconeVectorStore, QdrantVectorStore, WeaviateVectorStore
from dynamiq.types import Document
from dynamiq.utils.logger import logger


def get_documuments():
    documents = [
        Document(
            content="Kyiv is the best place to live in Ukraine",
            metadata={"city": "Kyiv"},
        ),
        Document(content="Kyiv is the capital of Ukraine", metadata={"city": "Kyiv"}),
        Document(
            content="Kyiv is the most populous city in Ukraine",
            metadata={"city": "Kyiv"},
        ),
        Document(
            content="Lviv is a city in western Ukraine", metadata={"city": "Lviv"}
        ),
        Document(
            content="Lviv is a city with a rich history", metadata={"city": "Lviv"}
        ),
        Document(
            content="Lviv is a city with a lot of tourists", metadata={"city": "Lviv"}
        ),
        Document(
            content="Odessa is a city in southern Ukraine", metadata={"city": "Odessa"}
        ),
        Document(
            content="Kharkiv is a great city in Ukraine", metadata={"city": "Kharkiv"}
        ),
        Document(
            content="Kharkiv is a city with a lot of students",
            metadata={"city": "Kharkiv"},
        ),
    ]

    return documents


def run_pinecone_indexing(documents):
    document_embedder_node = OpenAIDocumentEmbedder()

    document_writer_node = PineconeDocumentWriter(
        index_name="test-filtering",
        dimension=1536,
        depends=[
            NodeDependency(document_embedder_node),
        ],
        input_transformer=InputTransformer(
            selector={
                "documents": f"${[document_embedder_node.id]}.output.documents",
            },
        ),
    )

    indexing_flow = Flow(
        id="indexing_flow_pinecone_default",
        nodes=[
            document_embedder_node,
            document_writer_node,
        ],
        connection_manager=ConnectionManager(),
    )

    indexing_workflow = Workflow(id="indexing_workflow", flow=indexing_flow)

    result = indexing_workflow.run(
        input_data={
            "documents": documents,
        }
    )

    return result


def run_pinecone_retrieval():
    text_embedder_node = OpenAITextEmbedder()
    document_retriever_node = PineconeDocumentRetriever(
        index_name="test-filtering",
        dimension=1536,
        depends=[
            NodeDependency(text_embedder_node),
        ],
        input_transformer=InputTransformer(
            selector={
                "embedding": f"${[text_embedder_node.id]}.output.embedding",
            },
        ),
        filters={
            "operator": "OR",
            "conditions": [
                {"field": "city", "operator": "==", "value": "Kyiv"},
                {"field": "city", "operator": "==", "value": "Lviv"},
            ],
        },
    )

    retriever_flow = Flow(
        id="retriever_flow_pinecone_default",
        nodes=[
            text_embedder_node,
            document_retriever_node,
        ],
        connection_manager=ConnectionManager(),
    )

    retriever_worklow = Workflow(id="retriever_workflow", flow=retriever_flow)

    wf_output = retriever_worklow.run(
        input_data={
            "query": "Great city in Ukraine",
        }
    )

    output_documents = wf_output.output[document_retriever_node.id]["output"][
        "documents"
    ]

    pinecone_storage = PineconeVectorStore(index_name="test-filtering", dimension=1536)
    pinecone_storage.delete_documents(delete_all=True)

    logger.info(f"Output documents: {len(output_documents)}")

    assert len(output_documents) == 6

    return wf_output


def run_qdrant_indexing(documents):
    document_embedder_node = OpenAIDocumentEmbedder()

    document_writer_node = QdrantDocumentWriter(
        index_name="test-filtering",
        dimension=1536,
        depends=[
            NodeDependency(document_embedder_node),
        ],
        input_transformer=InputTransformer(
            selector={
                "documents": f"${[document_embedder_node.id]}.output.documents",
            },
        ),
    )

    indexing_flow = Flow(
        id="indexing_flow_qdrant_default",
        nodes=[
            document_embedder_node,
            document_writer_node,
        ],
        connection_manager=ConnectionManager(),
    )

    indexing_workflow = Workflow(id="indexing_workflow", flow=indexing_flow)

    result = indexing_workflow.run(
        input_data={
            "documents": documents,
        }
    )

    return result


def run_qdrant_retrieval():
    text_embedder_node = OpenAITextEmbedder()
    document_retriever_node = QdrantDocumentRetriever(
        index_name="test-filtering",
        dimension=1536,
        depends=[
            NodeDependency(text_embedder_node),
        ],
        input_transformer=InputTransformer(
            selector={
                "embedding": f"${[text_embedder_node.id]}.output.embedding",
            },
        ),
        filters={
            "operator": "OR",
            "conditions": [
                {"field": "city", "operator": "==", "value": "Kyiv"},
                {"field": "city", "operator": "==", "value": "Lviv"},
            ],
        },
    )

    retriever_flow = Flow(
        id="retriever_flow_qdrant_default",
        nodes=[
            text_embedder_node,
            document_retriever_node,
        ],
        connection_manager=ConnectionManager(),
    )

    retriever_worklow = Workflow(id="retriever_workflow", flow=retriever_flow)

    wf_output = retriever_worklow.run(
        input_data={
            "query": "Great city in Ukraine",
        }
    )

    output_documents = wf_output.output[document_retriever_node.id]["output"]["documents"]

    qdrant_storage = QdrantVectorStore(index_name="test-filtering", dimension=1536)
    qdrant_storage.delete_documents(delete_all=True)

    logger.info(f"Output documents: {len(output_documents)}")

    assert len(output_documents) == 6

    return wf_output


def run_weaviate_indexing(documents):
    document_embedder_node = OpenAIDocumentEmbedder()

    document_writer_node = WeaviateDocumentWriter(
        index_name="filtering",
        depends=[
            NodeDependency(document_embedder_node),
        ],
        input_transformer=InputTransformer(
            selector={
                "documents": f"${[document_embedder_node.id]}.output.documents",
            },
        ),
    )

    indexing_flow = Flow(
        id="indexing_flow_weaviate_default",
        nodes=[
            document_embedder_node,
            document_writer_node,
        ],
        connection_manager=ConnectionManager(),
    )

    indexing_workflow = Workflow(id="indexing_workflow", flow=indexing_flow)

    result = indexing_workflow.run(
        input_data={
            "documents": documents,
        }
    )

    return result


def run_weaviate_retrieval():
    text_embedder_node = OpenAITextEmbedder()
    document_retriever_node = WeaviateDocumentRetriever(
        index_name="filtering",
        depends=[
            NodeDependency(text_embedder_node),
        ],
        input_transformer=InputTransformer(
            selector={
                "embedding": f"${[text_embedder_node.id]}.output.embedding",
            },
        ),
        filters={
            "operator": "OR",
            "conditions": [
                {"field": "city", "operator": "==", "value": "Kyiv"},
                {"field": "city", "operator": "==", "value": "Lviv"},
            ],
        },
    )

    retriever_flow = Flow(
        id="retriever_flow_weaviate_default",
        nodes=[
            text_embedder_node,
            document_retriever_node,
        ],
        connection_manager=ConnectionManager(),
    )

    retriever_worklow = Workflow(id="retriever_workflow", flow=retriever_flow)

    wf_output = retriever_worklow.run(
        input_data={
            "query": "Great city in Ukraine",
        }
    )

    output_documents = wf_output.output[document_retriever_node.id]["output"][
        "documents"
    ]

    weaviate_storage = WeaviateVectorStore(index_name="filtering")
    weaviate_storage.delete_documents(delete_all=True)

    logger.info(f"Output documents: {len(output_documents)}")

    assert len(output_documents) == 6

    return wf_output


def run_chroma_indexing(documents):
    document_embedder_node = OpenAIDocumentEmbedder()

    document_writer_node = ChromaDocumentWriter(
        index_name="test-filtering",
        depends=[
            NodeDependency(document_embedder_node),
        ],
        input_transformer=InputTransformer(
            selector={
                "documents": f"${[document_embedder_node.id]}.output.documents",
            },
        ),
    )

    indexing_flow = Flow(
        id="indexing_flow_chroma_default",
        nodes=[
            document_embedder_node,
            document_writer_node,
        ],
        connection_manager=ConnectionManager(),
    )

    indexing_workflow = Workflow(id="indexing_workflow", flow=indexing_flow)

    result = indexing_workflow.run(
        input_data={
            "documents": documents,
        }
    )
    return result


def run_chroma_retrieval():
    text_embedder_node = OpenAITextEmbedder()
    document_retriever_node = ChromaDocumentRetriever(
        index_name="test-filtering",
        depends=[
            NodeDependency(text_embedder_node),
        ],
        input_transformer=InputTransformer(
            selector={
                "embedding": f"${[text_embedder_node.id]}.output.embedding",
            },
        ),
        filters={
            "operator": "OR",
            "conditions": [
                {"field": "city", "operator": "==", "value": "Kyiv"},
                {"field": "city", "operator": "==", "value": "Lviv"},
            ],
        },
    )

    retriever_flow = Flow(
        id="retriever_flow_chroma_default",
        nodes=[
            text_embedder_node,
            document_retriever_node,
        ],
        connection_manager=ConnectionManager(),
    )

    retriever_worklow = Workflow(id="retriever_workflow", flow=retriever_flow)

    wf_output = retriever_worklow.run(
        input_data={
            "query": "Great city in Ukraine",
        }
    )

    output_documents = wf_output.output[document_retriever_node.id]["output"][
        "documents"
    ]

    chroma_storage = ChromaVectorStore(index_name="test-filtering")
    chroma_storage.delete_documents(delete_all=True)

    logger.info(f"Output documents: {len(output_documents)}")

    assert len(output_documents) == 6

    return wf_output


def main():
    documents = get_documuments()
    run_pinecone_indexing(documents)
    run_qdrant_indexing(documents)
    run_weaviate_indexing(documents)
    run_chroma_indexing(documents)

    run_qdrant_retrieval()
    run_weaviate_retrieval()
    run_chroma_retrieval()
    run_pinecone_retrieval()


if __name__ == "__main__":
    main()
