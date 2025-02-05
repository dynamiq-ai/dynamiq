from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.agents.orchestrators.graph import END, START, GraphOrchestrator
from dynamiq.nodes.agents.orchestrators.graph_manager import GraphAgentManager
from dynamiq.nodes.llms import OpenAI
from examples.use_case_gpt_researcher.multi_agents import (
    plan_research,
    review_plan,
    run_initial_research,
    run_parallel_research,
    run_publisher,
    run_writer_agent,
)


def save_markdown_as_pdf(md_string: str, output_pdf: str):
    """Save a Markdown string as a PDF."""
    import markdown
    from weasyprint import HTML

    html_content = markdown.markdown(md_string)
    HTML(string=html_content).write_pdf(output_pdf)


def set_orchestrator() -> GraphOrchestrator:
    """Set up the orchestrator: multi-agent GPT-researcher."""
    llm = OpenAI(
        connection=OpenAIConnection(),
        model="gpt-4o-mini",
        temperature=0.1,
    )

    orchestrator = GraphOrchestrator(
        name="Graph orchestrator",
        manager=GraphAgentManager(llm=llm),
    )

    def orchestrate(context: dict, **kwargs) -> str:
        return (
            "researcher"
            if context["human_feedback"] is None or context["human_feedback"].lower().strip() == "no"
            else "planner"
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
    return orchestrator


if __name__ == "__main__":
    task = {
        "query": "Why is AI so hyped?",
        "max_sections": 1,
        "include_human_feedback": False,
        "follow_guidelines": True,
        "guidelines": [
            "The report MUST be written in APA format",
            "Each sub section MUST include supporting sources using hyperlinks."
            "If none exist, erase the sub section or rewrite it to be a part of the previous section",
        ],
        "source_to_extract": 5,
        "max_iterations": 1,
    }
    save_report_to_pdf = True

    orchestrator = set_orchestrator()
    orchestrator.context = {
        "task": task,
    }
    run_result = orchestrator.run(input_data={})
    report = orchestrator.context.get("report")

    if report:
        save_markdown_as_pdf(report, "report.pdf")
    print("Report:\n", report)
