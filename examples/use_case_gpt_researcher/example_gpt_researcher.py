import time

from examples.use_case_gpt_researcher.gpt_researcher.conduct_research import conduct_research_workflow
from examples.use_case_gpt_researcher.gpt_researcher.write_report import write_report_workflow
from examples.use_case_gpt_researcher.utils import save_markdown_as_pdf

if __name__ == "__main__":
    # If needed - clean Pinecone storage to remove old data
    # from examples.use_case_gpt_researcher.utils import clean_pinecone_storage
    # clean_pinecone_storage()

    task = {
        "query": "AI trends",  # Main topic query
        "num_sub_queries": 3,  # Number of sub-queries to expand search coverage
        "max_content_chunks_per_source": 2,  # Max number of content chunks to retrieve per URL from Pinecone
        "max_sources": 10,  # Max number of unique sources to include in the research
    }

    # Step 1: Conduct research and gather the relevant information
    conduct_research = conduct_research_workflow()
    conduct_research_result = conduct_research.run(
        input_data={
            "query": task.get("query"),
            "num_sub_queries": task.get("num_sub_queries"),
        }
    )

    # Step 2: Wait for Pinecone to load data to cloud before proceeding
    time.sleep(15)

    # Step 3: Generate the research report based on gathered information
    write_report = write_report_workflow(task.get("max_sources"))
    write_report_result = write_report.run(
        input_data={
            "query": task.get("query"),
            "max_sources": task.get("max_sources"),
            "max_content_per_source": task.get("max_content_chunks_per_source"),
        }
    )

    # Get the final research report
    report = write_report_result.output["generate_report_node"]["output"]["content"]

    if report:
        save_markdown_as_pdf(report, "report_gpt_researcher.pdf")

    print("Report:\n", report)
