import time

from examples.use_case_gpt_researcher.gpt_researcher import conduct_research_workflow, write_report_workflow


def run_initial_research(context: dict, **kwargs) -> dict:
    # Conduct research and gather the relevant information
    task = context.get("task")
    max_iterations = task.get("max_iterations")
    source_to_extract = task.get("source_to_extract")
    query = f'{task.get("query")} - {context.get("query")}' if context.get("query") else task.get("query")

    conduct_research = conduct_research_workflow()
    conduct_research.run(
        input_data={
            "query": query,
            "max_iterations": max_iterations,
        }
    )

    # Wait for Pinecone to load data to cloud before proceeding
    time.sleep(15)

    # Generate the research report based on gathered information
    write_report = write_report_workflow(source_to_extract)
    write_report_result = write_report.run(
        input_data={
            "query": query,
            "limit_sources": source_to_extract,
        }
    )

    # Get the final research report
    report = write_report_result.output["generate_report_node"]["output"]["content"]
    return {"initial_research": report, "result": "correct"}
