import os

import typer
from dotenv import find_dotenv, load_dotenv

from dynamiq import ROOT_PATH
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.flows import Flow
from dynamiq.nodes import InputTransformer, llms
from dynamiq.nodes.converters import UnstructuredFileConverter
from dynamiq.nodes.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from dynamiq.nodes.node import NodeDependency
from dynamiq.nodes.retrievers import PineconeDocumentRetriever
from dynamiq.nodes.splitters.document import DocumentSplitter
from dynamiq.nodes.writers import PineconeDocumentWriter
from dynamiq.prompts import Message, Prompt
from dynamiq.storages.vector.pinecone.pinecone import PineconeIndexType
from examples.rag.utils import list_data_folder_paths, read_bytes_io_files

load_dotenv(find_dotenv())

app = typer.Typer()


def create_indexing_flow(index_name="default"):
    # initialize indexing nodes
    file_converter_node = UnstructuredFileConverter(strategy="auto")
    document_splitter_node = DocumentSplitter(
        split_by="passage",
        depends=[
            NodeDependency(file_converter_node),
        ],
        input_transformer=InputTransformer(
            selector={
                "documents": f"${[file_converter_node.id]}.output.documents",
            },
        ),
    )
    document_embedder_node = OpenAIDocumentEmbedder(
        depends=[
            NodeDependency(document_splitter_node),
        ],
        input_transformer=InputTransformer(
            selector={
                "documents": f"${[document_splitter_node.id]}.output.documents",
            },
        ),
    )
    document_writer_node = PineconeDocumentWriter(
        index_name=index_name,
        index_type=PineconeIndexType.SERVERLESS,
        depends=[
            NodeDependency(document_embedder_node),
        ],
        input_transformer=InputTransformer(
            selector={
                "documents": f"${[document_embedder_node.id]}.output.documents",
            },
        ),
    )

    # build the indexing flow
    indexing_flow = Flow(
        id="indexing_flow_pinecone_default",
        nodes=[
            file_converter_node,
            document_splitter_node,
            document_embedder_node,
            document_writer_node,
        ],
    )
    return indexing_flow


@app.command()
def indexing_flow(
    folder_path: str = os.path.join(os.path.dirname(ROOT_PATH), "examples/data/"), index_name: str = "default"
):

    indexing_flow = create_indexing_flow(index_name=index_name)

    file_paths = list_data_folder_paths(folder_path)
    input_data = read_bytes_io_files(file_paths)

    # run the flow
    indexing_flow.run(
        input_data={
            "files": input_data["files"],
            "metadata": input_data["metadata"],
        }
    )


def default_prompt_template() -> str:
    """
    Returns the default prompt template for the language model.
    """
    return r"""
            Please answer the following question based on the information found
            within the sections enclosed by triplet quotes (\`\`\`).
            Your response should be concise, well-written, and follow markdown formatting guidelines:

            - Use bullet points for list items.
            - Use **bold** text for emphasis where necessary.

            **Question:** {{query}}

            Thank you for your detailed attention to the request
            **Context information**:
            ```
            {% for document in documents %}
                ---
                Document title: {{ document.metadata["title"] }}
                Document information: {{ document.content }}
                ---
            {% endfor %}
            ```

            **User Question:** {{query}}
            Answer:
            """


def create_retrieval_flow(index_name: str = "default"):
    # initialize the retriver nodes
    text_embedder_node = OpenAITextEmbedder()
    document_retriever_node = PineconeDocumentRetriever(
        index_name=index_name,
        depends=[
            NodeDependency(text_embedder_node),
        ],
        input_transformer=InputTransformer(
            selector={
                "embedding": f"${[text_embedder_node.id]}.output.embedding",
            },
        ),
    )

    # intitalize the llm exectutor node
    message = Message(content=default_prompt_template(), role="user")
    prompt = Prompt(id="1", messages=[message])

    answer_generation_node = llms.OpenAI(
        id="answer_generation_node_id_1",
        name="OpenAI Answer Generation",
        model="gpt-3.5-turbo",
        prompt=prompt,
        connection=OpenAIConnection(),
        depends=[
            NodeDependency(document_retriever_node),
        ],
        input_transformer=InputTransformer(
            selector={
                "query": "$.query",
                "documents": f"${[document_retriever_node.id]}.output.documents",
            },
        ),
    )

    # build the retrieval flow
    retriever_flow = Flow(
        id="retriever_flow_pinecone_default",
        nodes=[
            text_embedder_node,
            document_retriever_node,
            answer_generation_node,
        ],
    )
    return retriever_flow


@app.command()
def retrieval_flow(
    question: str = "How to build an advanced RAG pipeline?",
    index_name: str = "default",
):

    retriever_flow = create_retrieval_flow(index_name=index_name)
    answer_generation_node_id = "answer_generation_node_id_1"

    flow_result = retriever_flow.run(input_data={"query": question})
    answer = (
        flow_result.output.get(answer_generation_node_id).get("output").get("content")
    )
    print("Answer:\n", answer)


@app.command()
def main(
    folder_path=os.path.join(os.path.dirname(ROOT_PATH), "examples/data/"),
    question="How to build an advanced RAG pipeline?",
):
    indexing_flow(folder_path)
    retrieval_flow(question)


if __name__ == "__main__":
    main()
