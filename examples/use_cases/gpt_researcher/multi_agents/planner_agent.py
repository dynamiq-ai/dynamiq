from datetime import datetime

from examples.use_cases.gpt_researcher.multi_agents.utils import execute_agent


def _plan_research_prompt(
    initial_research: str, include_human_feedback: bool, human_feedback: str, max_sections: int
) -> str:
    """Generate a research prompt with optional human feedback."""
    feedback_instruction = (
        f"Human feedback: {human_feedback}. You must plan the sections based on the human feedback."
        if include_human_feedback and human_feedback and human_feedback.lower() != "no"
        else ""
    )

    return f"""Today's date is {datetime.now().strftime('%d/%m/%Y')}
Research summary report: '{initial_research}'
{feedback_instruction}

\nYour task is to generate an outline of sections headers for the research project
based on the research summary report above.
You must generate a maximum of {max_sections} section headers.
You must focus ONLY on related research topics for subheaders and do NOT include introduction,
conclusion and references.
You must return nothing but a JSON with the fields 'title' (str) and
'sections' (maximum {max_sections} section headers) with the following structure:
'{{title: string research title, date: today's date,
sections: ['section header 1', 'section header 2', 'section header 3' ...]}}'."""


def plan_research(context: dict, **kwargs) -> dict:
    """Main function to plan the research layout based on the context provided."""
    initial_research = context.get("initial_research")
    human_feedback = context.get("human_feedback")
    task = context.get("task")

    include_human_feedback = task.get("include_human_feedback", False)
    max_sections = task.get("max_sections", 5)

    system_prompt = (
        "You are a research editor. Your goal is to oversee the research project from inception to completion. "
        "Your main task is to plan the article section layout based on an initial research summary."
    )
    user_prompt = _plan_research_prompt(initial_research, include_human_feedback, human_feedback, max_sections)

    response = execute_agent(system_prompt, user_prompt, to_json=True)
    return {
        "title": response.get("title"),
        "date": response.get("date"),
        "sections": response.get("sections", []),
        "result": "",
    }
