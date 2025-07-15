import logging

from dynamiq import Workflow, runnables
from dynamiq.connections import connections
from dynamiq.connections.managers import ConnectionManager
from dynamiq.flows import Flow
from dynamiq.nodes.embedders import (
    BedrockDocumentEmbedder,
    BedrockTextEmbedder,
    CohereDocumentEmbedder,
    CohereTextEmbedder,
    HuggingFaceDocumentEmbedder,
    HuggingFaceTextEmbedder,
    MistralDocumentEmbedder,
    MistralTextEmbedder,
)
from dynamiq.types import Document


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    cm = ConnectionManager()

    mistral_connection = connections.Mistral()
    cohere_connection = connections.Cohere()
    huggingface_connection = connections.HuggingFace()
    bedrock_connection = connections.AWS(profile="")

    mistral_text_embedder_node = MistralTextEmbedder(name="MistralTextEmbedder", connection=mistral_connection)
    cohere_text_embedder_node = CohereTextEmbedder(name="CohereTextEmbedder", connection=cohere_connection)
    huggingface_text_embedder_node = HuggingFaceTextEmbedder(
        name="HuggingFaceTextEmbedder", connection=huggingface_connection
    )
    bedrock_text_embedder_node = BedrockTextEmbedder(name="BedrockTextEmbedder", connection=bedrock_connection)

    mistral_document_embedder_node = MistralDocumentEmbedder(
        name="MistralDocumentEmbedder", connection=mistral_connection
    )
    cohere_document_embedder_node = CohereDocumentEmbedder(name="CohereDocumentEmbedder", connection=cohere_connection)
    huggingface_document_embedder_node = HuggingFaceDocumentEmbedder(
        name="HuggingFaceDocumentEmbedder", connection=huggingface_connection
    )
    bedrock_document_embedder_node = BedrockDocumentEmbedder(
        name="BedrockDocumentEmbedder", connection=bedrock_connection
    )

    text_nodes = [
        mistral_text_embedder_node,
        cohere_text_embedder_node,
        huggingface_text_embedder_node,
        bedrock_text_embedder_node,
    ]

    document_nodes = [
        mistral_document_embedder_node,
        cohere_document_embedder_node,
        huggingface_document_embedder_node,
        bedrock_document_embedder_node,
    ]
    for node in text_nodes:
        wf = Workflow(
            id="wf",
            flow=Flow(
                id="wf",
                nodes=[node],
                connection_manager=cm,
            ),
        )
        response = wf.run(
            input_data={"query": "Embedder executing"},
            config=runnables.RunnableConfig(callbacks=[]),
        )
        logger.info(f"Workflow result for {node.name}:{response}")

    for node in document_nodes:
        wf = Workflow(
            id="wf",
            flow=Flow(
                id="wf",
                nodes=[node],
                connection_manager=cm,
            ),
        )
        response = wf.run(
            input_data={"documents": [Document(content="Embedder executing")]},
            config=runnables.RunnableConfig(callbacks=[]),
        )
        logger.info(f"Workflow result for {node.name}:{response.output}")


if __name__ == "__main__":
    main()
