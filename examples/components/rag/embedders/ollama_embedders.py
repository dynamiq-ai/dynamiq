from dynamiq.connections import Ollama as OllamaConnection
from dynamiq.nodes.embedders import OllamaDocumentEmbedder, OllamaTextEmbedder
from dynamiq.types import Document

# Pull the model first with `ollama pull nomic-embed-text`.
MODEL = "ollama/nomic-embed-text"


def run_ollama_embedders(connection: OllamaConnection):
    text_embedder = OllamaTextEmbedder(connection=connection, model=MODEL)
    document_embedder = OllamaDocumentEmbedder(connection=connection, model=MODEL)

    text_result = text_embedder.run(input_data={"query": "What is entropy?"})
    document_result = document_embedder.run(
        input_data={
            "documents": [
                Document(content="Entropy measures the disorder of a system."),
                Document(content="Enthalpy is the total heat content of a system."),
            ]
        }
    )
    return text_result, document_result


if __name__ == "__main__":
    text_result, document_result = run_ollama_embedders(OllamaConnection(url="http://localhost:11434"))
    print(f"Query embedding size: {len(text_result.output['embedding'])}")
    for document in document_result.output["documents"]:
        print(f"{document.content[:40]!r}: {len(document.embedding)} dimensions")
