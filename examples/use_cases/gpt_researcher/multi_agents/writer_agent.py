import json
from datetime import datetime

from examples.use_cases.gpt_researcher.multi_agents.utils import execute_agent


def run_writer_agent(context: dict, **kwargs) -> dict:
    """
    Orchestrates the research writing process by generating sections content and revising headers.
    """
    return {**_generate_writer_sections(context), **_revise_headers(context), "result": "success"}


def _generate_writer_sections(context: dict) -> dict:
    """
    Generates the introduction, conclusion, and table of contents based on research data.
    """
    query = context.get("title")
    data = context.get("research_data")
    task = context.get("task")

    follow_guidelines = task.get("follow_guidelines")
    guidelines = task.get("guidelines")

    user_prompt = f"""
Today's date is {datetime.now().strftime('%d/%m/%Y')}.
Query or Topic: {query}
Research data: {str(data)}

Your task is to write an in-depth, well-written, and detailed introduction and conclusion
for the research report based on the provided data.

Guidelines:
- Do NOT include headers in the results.
- Include relevant sources as markdown hyperlinks (e.g., 'This is a sample text. ([source](url))').
- {f"Follow these guidelines: {guidelines}" if follow_guidelines else "No additional guidelines."}
- Return ONLY a JSON response in the following format:

{{
    "table_of_contents": "- A table of contents in markdown syntax based on the research data.",
    "introduction": "An in-depth introduction with markdown formatting and hyperlink references.",
    "conclusion": "A conclusion summarizing the research with markdown formatting and hyperlinks.",
    "sources": [
        "- Title, Year, Author [source](url)",
        ...
    ]
}}"""

    system_prompt = "You are a research assistant. Your task is to process and generate structured research content."

    result = execute_agent(system_prompt, user_prompt, to_json=True)

    return {
        "table_of_contents": result.get("table_of_contents", None),
        "introduction": result.get("introduction", None),
        "conclusion": result.get("conclusion", None),
        "sources": result.get("sources", []),
    }


def _revise_headers(context: dict) -> dict:
    """
    Revises headers based on given guidelines.
    """
    task = context.get("task")

    default_headers = {
        "title": context.get("title"),
        "date": "Date",
        "introduction": "Introduction",
        "table_of_contents": "Table of Contents",
        "conclusion": "Conclusion",
        "references": "References",
    }

    user_prompt = f"""
Your task is to revise the given headers JSON based on the guidelines provided.
- The values should be in simple strings, ignoring all markdown syntax.
- You must return a JSON in the same format as the headers data.

Guidelines: {task.get("guidelines", "No specific guidelines.")}

Headers Data: {json.dumps(default_headers, indent=2)}
    """

    system_prompt = "You are a research assistant. Your task is to process and generate structured research content."

    result = execute_agent(system_prompt, user_prompt, to_json=True)
    return {
        "headers": result,
    }
