import typer

from dynamiq.connections import Firecrawl as FirecrawlConnection
from dynamiq.flows import Flow
from dynamiq.nodes import InputTransformer
from dynamiq.nodes.embedders import OpenAIDocumentEmbedder
from dynamiq.nodes.node import NodeDependency
from dynamiq.nodes.splitters.document import DocumentSplitter
from dynamiq.nodes.tools.firecrawl import FirecrawlTool
from dynamiq.nodes.tools.python import Python
from dynamiq.nodes.writers import PineconeDocumentWriter
from dynamiq.storages.vector import PineconeVectorStore

app = typer.Typer()

python_code = """
def run(inputs):
    from dynamiq.types import Document

    raw_content = input_data.get('content')
    content = raw_content.get('markdown')

    document = Document(content=content)
    return {
        'documents': [document,]
    }
"""


def create_indexing_flow(index_name="default"):
    vector_store = PineconeVectorStore(index_name=index_name, dimension=1536)

    web_scraper_tool = FirecrawlTool(connection=FirecrawlConnection())

    python_node = Python(
        code=python_code,
        depends=[
            NodeDependency(web_scraper_tool),
        ],
        input_transformer=InputTransformer(
            selector={
                "content": f"${[web_scraper_tool.id]}.output.content",
            },
        ),
    )

    document_splitter_node = DocumentSplitter(
        split_by="passage",
        depends=[
            NodeDependency(python_node),
        ],
        input_transformer=InputTransformer(
            selector={
                "documents": f"${[python_node.id]}.output.documents",
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
        vector_store=vector_store,
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
            web_scraper_tool,
            python_node,
            document_splitter_node,
            document_embedder_node,
            document_writer_node,
        ],
    )
    return indexing_flow


@app.command()
def indexing_flow(url: str = "https://thedubaimall.com/", index_name: str = "default"):

    indexing_flow = create_indexing_flow(index_name=index_name)

    # run the flow
    output = indexing_flow.run(
        input_data={
            "url": url,
        }
    )

    return output


@app.command()
def main(
    url="https://thedubaimall.com/",
):
    indexing_flow(url)


if __name__ == "__main__":
    indexing_flow("https://thedubaimall.com/")
