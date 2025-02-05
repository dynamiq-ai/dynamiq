from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.agents.orchestrators.graph import END, START, GraphOrchestrator
from dynamiq.nodes.agents.orchestrators.graph_manager import GraphAgentManager
from dynamiq.nodes.llms import OpenAI
from examples.use_case_gpt_researcher.multi_agents.researcher_agent import run_initial_research
from examples.use_case_gpt_researcher.multi_agents.utils import execute_llm


def run_parallel_research(context: dict) -> dict:
    """
    Runs research for multiple queries and aggregates the results.
    """
    queries = context.get("sections", [])
    research_data = [editor_agent(context, query).get("draft", "") for query in queries]
    return {
        "result": "success",
        "research_data": research_data,
    }


def editor_agent(context: dict, query: str) -> dict:
    """
    Runs a research orchestration process for a given query.
    """
    llm = OpenAI(
        connection=OpenAIConnection(),
        model="gpt-4o-mini",
        temperature=0.1,
    )

    orchestrator = GraphOrchestrator(
        name="Reviser graph orchestrator",
        manager=GraphAgentManager(llm=llm),
    )

    def orchestrate(context: dict, **kwargs) -> str:
        return END if context["review"] is None else "reviser"

    orchestrator.add_state_by_tasks("researcher", [_run_in_depth_research])
    orchestrator.add_state_by_tasks("reviewer", [_review_draft])
    orchestrator.add_state_by_tasks("reviser", [_revise_draft])

    orchestrator.add_edge(START, "researcher")
    orchestrator.add_edge("researcher", "reviewer")
    orchestrator.add_conditional_edge("reviewer", ["reviser", END], orchestrate)
    orchestrator.add_edge("reviser", END)

    orchestrator.context = {
        "query": query,
        "task": context.get("task"),
    }
    orchestrator.run(input_data={})
    return {"draft": orchestrator.context.get("draft"), "result": ""}


def _run_in_depth_research(context: dict) -> dict:
    """
    Run GPT Researcher on the given query.
    """
    result = run_initial_research(context)
    report = result.get("initial_research")
    return {"draft": report, "result": ""}


def _review_draft(context: dict) -> dict:
    """
    Reviews the draft based on guidelines and determines if revisions are needed.
    """
    task = context.get("task")
    if not task.get("follow_guidelines", False):
        return {"review": None, "result": ""}

    guidelines = task.get("guidelines")
    revision_notes = context.get("revision_notes")
    draft = context.get("draft")

    revise_prompt = f"""The reviser has already revised the draft based on your previous review notes with
the following feedback: {revision_notes}\n
Please provide additional feedback ONLY if critical since the reviser has already made changes based
on your previous feedback.
If you think the article is sufficient or that non critical revisions are required, please aim to return None."""

    review_prompt = f"""You have been tasked with reviewing the draft which was written by a non-expert based on
specific guidelines.
Please accept the draft if it is good enough to publish, or send it for revision, along with your notes to guide the
revision.
If not all of the guideline criteria are met, you should send appropriate revision notes.
If the draft meets all the guidelines, please return None.
{revise_prompt if revision_notes else ""}

Guidelines: {guidelines}\nDraft: {draft}\n
"""

    system_prompt = """You are an expert research article reviewer.\
    Your goal is to review research drafts and provide feedback to the reviser only based on specific guidelines."""

    response = execute_llm(system_prompt, review_prompt)
    return {"review": response, "result": ""}


def _revise_draft(context: dict) -> dict:
    """
    Revises the draft based on reviewer feedback.
    """
    review = context.get("review")
    draft_report = context.get("draft")

    user_prompt = f"""Draft:\n{draft_report}" + "Reviewer's notes:\n{review}\n\n
You have been tasked by your reviewer with revising the following draft, which was written by a non-expert.
If you decide to follow the reviewer's notes,
please write a new draft and make sure to address all of the points they raised.
Please keep all other aspects of the draft the same.
You MUST return nothing but a JSON in the following format:

{{
    "draft": The revised draft that you are submitting for review,
    "revision_notes": Your message to the reviewer about the changes you made to the draft based on their feedback
}}
"""
    system_prompt = "You are an expert writer. Your goal is to revise drafts based on reviewer notes."

    response = execute_llm(system_prompt, user_prompt, to_json=True)

    return {"draft": response.get("draft", ""), "revision_notes": response.get("revision_notes", ""), "result": ""}
