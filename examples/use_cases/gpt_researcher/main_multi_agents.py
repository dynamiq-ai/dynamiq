from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes import ErrorHandling
from dynamiq.nodes.agents.orchestrators.graph import END, START, GraphOrchestrator
from dynamiq.nodes.agents.orchestrators.graph_manager import GraphAgentManager
from dynamiq.nodes.llms import OpenAI
from examples.use_cases.gpt_researcher.multi_agents import (
    plan_research,
    review_plan,
    run_initial_research,
    run_parallel_research,
    run_publisher,
    run_writer_agent,
)
from examples.use_cases.gpt_researcher.utils import save_markdown_as_pdf


def set_orchestrator() -> GraphOrchestrator:
    """Set up the orchestrator: multi-agent GPT-researcher."""

    def orchestrate(context: dict, **kwargs) -> str:
        human_feedback = context.get("human_feedback")
        return "researcher" if human_feedback is None or human_feedback.lower().strip() == "no" else "planner"

    llm = OpenAI(
        connection=OpenAIConnection(),
        model="gpt-4o-mini",
        temperature=0.1,
    )

    orchestrator = GraphOrchestrator(
        name="Graph orchestrator",
        manager=GraphAgentManager(llm=llm),
    )

    orchestrator.add_state_by_tasks("browser", [run_initial_research])
    orchestrator.add_state_by_tasks("planner", [plan_research])
    orchestrator.add_state_by_tasks("human", [review_plan])
    orchestrator.add_state_by_tasks("researcher", [run_parallel_research])
    orchestrator.add_state_by_tasks("writer", [run_writer_agent])
    orchestrator.add_state_by_tasks("publisher", [run_publisher])

    orchestrator.add_edge(START, "browser")
    orchestrator.add_edge("browser", "planner")
    orchestrator.add_edge("planner", "human")
    orchestrator.add_conditional_edge("human", ["planner", "researcher"], orchestrate)
    orchestrator.add_edge("researcher", "writer")
    orchestrator.add_edge("writer", "publisher")
    orchestrator.add_edge("publisher", END)

    for i in range(len(orchestrator.states)):
        if orchestrator.states[i].id not in ["START", "END"]:
            orchestrator.states[i].tasks[0].error_handling = ErrorHandling(timeout_seconds=None)

    return orchestrator


if __name__ == "__main__":
    # If needed - clean Pinecone storage to remove old data
    # from examples.use_case_gpt_researcher.utils import clean_pinecone_storage
    # clean_pinecone_storage()

    task = {
        "query": "AI trends",  # Main topic query
        "num_sub_queries": 3,  # Number of sub-queries to expand search coverage
        "max_content_chunks_per_source": 5,  # Max number of content chunks to retrieve per URL from Pinecone
        "max_sources": 10,  # Max number of unique sources per section to include in the research
        "max_sections": 5,  # Max number of sections in the research
        "include_human_feedback": False,  # Adjust section topics based on user feedback
        "follow_guidelines": True,  # Apply additional guidelines to LLM instructions
        "guidelines": [
            "The report MUST be written in APA format",
            "Each sub section MUST include supporting sources using hyperlinks."
            "If none exist, erase the sub section or rewrite it to be a part of the previous section",
        ],
    }

    orchestrator = set_orchestrator()
    orchestrator.context = {
        "task": task,
    }
    orchestrator.run(input_data={})

    report = orchestrator.context.get("report")

    if report:
        save_markdown_as_pdf(report, "report_multi_agents.pdf")

    print("Report:\n", report)
