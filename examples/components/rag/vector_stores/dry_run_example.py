from dotenv import load_dotenv

from dynamiq import Workflow
from dynamiq.connections import Weaviate
from dynamiq.flows import Flow
from dynamiq.nodes.operators import Map
from dynamiq.nodes.writers.weaviate import WeaviateDocumentWriter
from dynamiq.runnables import RunnableConfig
from dynamiq.types import Document
from dynamiq.types.dry_run import DryRunConfig


def weaviate_dry_run():
    load_dotenv()

    # Create sample documents
    documents = [
        Document(id="doc1", content="Sample content 1", embedding=[0.1, 0.2, 0.3], metadata={"name": "doc1"}),
        Document(id="doc2", content="Sample content 2", embedding=[0.4, 0.5, 0.6], metadata={"name": "doc2"}),
        Document(id="doc3", content="Sample content 3", embedding=[0.7, 0.8, 0.9], metadata={"name": "doc3"}),
    ]

    # Create runnable config with dry run
    dry_run_config = DryRunConfig(enabled=True, delete_collection=False, delete_documents=True)
    config = RunnableConfig(dry_run=dry_run_config)

    # Add a writer node
    writer_node = WeaviateDocumentWriter(
        connection=Weaviate(),
        index_name="Test_dry_run",
        create_if_not_exist=True,
    )

    # Add a map node
    map_node = Map(
        node=writer_node,
    )

    # Create a workflow with the map node
    workflow = Workflow(flow=Flow(nodes=[map_node]))

    inputs = {"input": [{"documents": [documents[0]]}, {"documents": [documents[1]]}, {"documents": [documents[2]]}]}

    # Run workflow with dry run config
    result = workflow.run(
        input_data=inputs,
        config=config,
    )

    return result


if __name__ == "__main__":
    weaviate_dry_run()
