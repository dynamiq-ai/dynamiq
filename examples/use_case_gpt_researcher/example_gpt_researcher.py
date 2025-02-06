import time

from dynamiq.connections import Pinecone
from dynamiq.storages.vector.pinecone import PineconeVectorStore
from examples.use_case_gpt_researcher.gpt_researcher.conduct_research import conduct_research_workflow
from examples.use_case_gpt_researcher.gpt_researcher.write_report import write_report_workflow


def clean_pinecone_storage():
    """Deletes all documents in the Pinecone vector storage."""
    vector_store = PineconeVectorStore(connection=Pinecone(), index_name="gpt-researcher", create_if_not_exist=True)

    vector_store.delete_documents(delete_all=True)


if __name__ == "__main__":
    task = {
        "query": "Ai trends",
        "max_iterations": 2,
        "source_to_extract": 20,
        "limit_sources": 10,
    }

    # If needed - clean Pinecone storage to remove old data
    # clean_pinecone_storage()

    # Step 1: Conduct research and gather the relevant information
    conduct_research = conduct_research_workflow()
    conduct_research_result = conduct_research.run(
        input_data={
            "query": task.get("query"),
            "max_iterations": task.get("max_iterations"),
        }
    )

    # Step 2: Wait for Pinecone to load data to cloud before proceeding
    time.sleep(15)

    # Step 3: Generate the research report based on gathered information
    write_report = write_report_workflow(task.get("source_to_extract"))
    write_report_result = write_report.run(
        input_data={
            "query": task.get("query"),
            "limit_sources": task.get("limit_sources"),
        }
    )

    # Get the final research report
    print(write_report_result.output["generate_report_node"]["output"]["content"])
