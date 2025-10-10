"""Job posting example using a manager agent with specialized sub-agents as tools."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable

from dynamiq import Workflow
from dynamiq.connections import ScaleSerp
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.tools.scale_serp import ScaleSerpTool
from dynamiq.nodes.types import Behavior, InferenceMode
from dynamiq.utils.logger import logger

from examples.components.tools.custom_tools.file_reader import FileReaderTool
from examples.llm_setup import setup_llm

OUTPUT_FILE_PATH = Path("job_posting.md")
DEFAULT_BRIEF_PATH = Path("job_example.md")


def _optional_file_reader(file_path: Path | None) -> FileReaderTool | None:
    if not file_path:
        return None

    if not file_path.exists():
        logger.warning("Company brief path %s does not exist; skipping file reader tool.", file_path)
        return None

    content = file_path.read_bytes()
    buffer = io.BytesIO(content)
    buffer.description = file_path.name  # type: ignore[attr-defined]
    return FileReaderTool(files=[buffer])


def _research_agent(llm, search_tool: ScaleSerpTool, file_reader: FileReaderTool | None) -> Agent:
    role = (
        "You are the Research Strategist sub-agent.\n"
        "- Always expect tool input as {'input': '<research task>'}.\n"
        "- Use web search to gather company positioning, market facts, and talent expectations.\n"
        "- If a company brief is available, call the file reader to extract tone, mission, and benefits.\n"
        "- Return concise findings with bullet points and source notes the writer can reference."
    )
    tools = [search_tool]
    if file_reader:
        tools.append(file_reader)

    return Agent(
        name="Research Strategist",
        description="Compiles hiring insights using search and optional company brief.",
        role=role,
        llm=llm,
        tools=tools,
        inference_mode=InferenceMode.XML,
        parallel_tool_calls_enabled=True,
        max_loops=8,
        behaviour_on_max_loops=Behavior.RETURN,
    )


def _value_prop_agent(llm, file_reader: FileReaderTool | None) -> Agent:
    role = (
        "You are the Employer Value Proposition sub-agent.\n"
        "- Always expect tool input as {'input': '<context>'}.\n"
        "- Distill the unique culture, growth story, and benefits into punchy messaging.\n"
        "- If the company brief is available, ground your messaging in real details.\n"
        "- Deliver three talking points plus a short elevator pitch tone guide."
    )
    tools = [file_reader] if file_reader else []

    return Agent(
        name="Value Proposition Curator",
        description="Shapes employer messaging from context and briefs.",
        role=role,
        llm=llm,
        tools=tools,
        inference_mode=InferenceMode.XML,
        max_loops=6,
        behaviour_on_max_loops=Behavior.RETURN,
    )


def _writer_agent(llm, file_reader: FileReaderTool | None) -> Agent:
    role = (
        "You are the Job Posting Writer sub-agent.\n"
        "- Always expect tool input as {'input': '<draft context>'}.\n"
        "- Produce a markdown job post with sections: Overview, Responsibilities, Must-Haves, Nice-to-Haves, Benefits, How to Apply.\n"
        "- Mirror the company's tone and weave in the value proposition.\n"
        "- Cite any external sources in footnotes if included."
    )
    tools = [file_reader] if file_reader else []

    return Agent(
        name="Job Posting Writer",
        description="Drafts the job post in structured markdown.",
        role=role,
        llm=llm,
        tools=tools,
        inference_mode=InferenceMode.XML,
        max_loops=6,
        behaviour_on_max_loops=Behavior.RETURN,
    )


def _editor_agent(llm) -> Agent:
    role = (
        "You are the Hiring Communications Editor sub-agent.\n"
        "- Always expect tool input as {'input': '<draft>'}.\n"
        "- Review for clarity, inclusive language, and signal-to-noise ratio.\n"
        "- Ensure the call-to-action is actionable and highlight key perks.\n"
        "- Return an edited markdown draft plus a changelog of improvements."
    )
    return Agent(
        name="Communications Editor",
        description="Refines the job post for clarity and impact.",
        role=role,
        llm=llm,
        inference_mode=InferenceMode.XML,
        max_loops=6,
        behaviour_on_max_loops=Behavior.RETURN,
    )


def _manager_agent(llm, subagents: Iterable[Agent]) -> Agent:
    role = (
        "You are the Job Campaign Manager coordinating recruiting specialists.\n"
        "- Break the request into research, messaging, drafting, and editing tasks.\n"
        "- When invoking a sub-agent tool ALWAYS pass {'input': '<task>'}.\n"
        "- Favor parallel calls when research and messaging can progress independently.\n"
        "- Deliver a polished markdown job posting aligned with the provided constraints."
    )
    return Agent(
        name="Job Campaign Manager",
        description="Orchestrates sub-agents to produce the final job posting.",
        role=role,
        llm=llm,
        tools=list(subagents),
        inference_mode=InferenceMode.XML,
        parallel_tool_calls_enabled=True,
        max_loops=12,
        behaviour_on_max_loops=Behavior.RETURN,
    )


def _build_llm(model_provider: str | None = None, model_name: str | None = None):
    kwargs = {}
    if model_provider:
        kwargs["model_provider"] = model_provider
    if model_name:
        kwargs["model_name"] = model_name
    return setup_llm(**kwargs)


def run_job_posting(
    input_data: dict,
    company_brief_path: Path | None = DEFAULT_BRIEF_PATH,
    model_provider: str | None = None,
    model_name: str | None = None,
) -> dict:
    """Execute the job posting workflow and return the manager agent output."""
    llm = _build_llm(model_provider=model_provider, model_name=model_name)
    search_tool = ScaleSerpTool(connection=ScaleSerp())
    file_reader = _optional_file_reader(company_brief_path)

    researcher = _research_agent(llm, search_tool, file_reader)
    value_prop = _value_prop_agent(llm, file_reader)
    writer = _writer_agent(llm, file_reader)
    editor = _editor_agent(llm)
    manager = _manager_agent(llm, [researcher, value_prop, writer, editor])

    workflow = Workflow(flow=Flow(nodes=[manager]))

    prompt = (
        "Create a recruiting-ready job posting. Incorporate company tone, highlight signature benefits, "
        "and ensure the role description is actionable for senior talent.\n\n"
        f"Company website or link: {input_data.get('company_link', 'N/A')}\n"
        f"Company domain / industry: {input_data.get('company_domain', 'N/A')}\n"
        f"Role hiring for: {input_data.get('hiring_needs', 'N/A')}\n"
        f"Key benefits to emphasize: {input_data.get('specific_benefits', 'N/A')}\n"
    )

    tone = input_data.get("tone")
    must_have = input_data.get("must_have_points")
    nice_to_have = input_data.get("nice_to_have_points")

    if tone:
        prompt += f"Preferred tone: {tone}\n"
    if must_have:
        prompt += f"Must include: {must_have}\n"
    if nice_to_have:
        prompt += f"Nice-to-have accents: {nice_to_have}\n"

    prompt += "\nReturn the final job post in markdown with footnotes for any external references."

    result = workflow.run(input_data={"input": prompt})
    logger.info("Job posting workflow completed")
    return result.output.get(manager.id, {})


def main() -> None:
    company_link = input("Company website or link: ")
    company_domain = input("Company domain / industry: ")
    hiring_needs = input("Role hiring for: ")
    benefits = input("Key benefits or perks to highlight: ")
    tone = input("Preferred tone (optional): ")
    must_have = input("Must-have talking points (optional): ")
    nice_to_have = input("Nice-to-have accents (optional): ")
    brief_path_input = input(f"Path to company brief markdown (press Enter for {DEFAULT_BRIEF_PATH}): ")

    brief_path = None
    if brief_path_input.strip():
        brief_path = Path(brief_path_input.strip())
    else:
        brief_path = DEFAULT_BRIEF_PATH if DEFAULT_BRIEF_PATH.exists() else None

    payload = {
        "company_link": company_link,
        "company_domain": company_domain,
        "hiring_needs": hiring_needs,
        "specific_benefits": benefits,
        "tone": tone or None,
        "must_have_points": must_have or None,
        "nice_to_have_points": nice_to_have or None,
    }

    manager_output = run_job_posting(payload, company_brief_path=brief_path)
    content = manager_output.get("output", {}).get("content")

    if content:
        print(content)
        OUTPUT_FILE_PATH.write_text(content)
        print(f"\nSaved job posting to {OUTPUT_FILE_PATH.resolve()}")
    else:
        print("Workflow completed without textual content. Raw output:")
        print(manager_output)


if __name__ == "__main__":
    main()
