"""Literature overview example using a manager agent with specialized researcher sub-agents."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from dynamiq import Workflow
from dynamiq.connections import ScaleSerp, ZenRows
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.tools.scale_serp import ScaleSerpTool
from dynamiq.nodes.types import Behavior, InferenceMode
from dynamiq.utils.logger import logger

from examples.components.tools.custom_tools.scraper import ScraperSummarizerTool
from examples.llm_setup import setup_llm

OUTPUT_FILE_PATH = Path("literature_overview.md")


def _research_agent(llm, search_tool: ScaleSerpTool, scraper_tool: ScraperSummarizerTool) -> Agent:
    role = (
        "You are the Lead Research Analyst sub-agent.\n"
        "- Always expect tool input as {'input': '<topic or question>'}.\n"
        "- Use the web search and scraper tools to surface current, credible information.\n"
        "- Provide key findings with inline source references and include the raw URLs you touched."
    )
    return Agent(
        name="Lead Research Analyst",
        description="Surfaces fresh evidence using search + scraping tools.",
        role=role,
        llm=llm,
        tools=[search_tool, scraper_tool],
        inference_mode=InferenceMode.FUNCTION_CALLING,
        parallel_tool_calls_enabled=True,
        max_loops=8,
        behaviour_on_max_loops=Behavior.RETURN,
    )


def _citation_agent(llm) -> Agent:
    role = (
        "You are the Citation Curator sub-agent.\n"
        "- Always expect tool input as {'input': '<research summary>'}.\n"
        "- Extract a clean bibliography with annotated highlights for each source.\n"
        "- Return structured bullet points that the writer can reference directly."
    )
    return Agent(
        name="Citation Curator",
        description="Normalizes research into a citation digest.",
        role=role,
        llm=llm,
        inference_mode=InferenceMode.FUNCTION_CALLING,
        max_loops=6,
        behaviour_on_max_loops=Behavior.RETURN,
    )


def _writer_agent(llm) -> Agent:
    role = (
        "You are the Literature Writer sub-agent.\n"
        "- Always expect tool input as {'input': '<synthesis context>'}.\n"
        "- Produce a markdown literature overview with introduction, key themes, applications, and conclusion.\n"
        "- Cite sources inline using markdown footnotes and include a final references list."
    )
    return Agent(
        name="Literature Writer",
        description="Delivers the polished overview in markdown format.",
        role=role,
        llm=llm,
        inference_mode=InferenceMode.FUNCTION_CALLING,
        max_loops=6,
        behaviour_on_max_loops=Behavior.RETURN,
    )


def _manager_agent(llm, subagents: Iterable[Agent]) -> Agent:
    role = (
        "You are the Literature Overview Manager coordinating researcher sub-agents.\n"
        "- Break the user request into clear subtasks and delegate via the available tools.\n"
        "- When invoking a sub-agent tool ALWAYS pass {'input': '<task>'}.\n"
        "- Encourage parallel calls when multiple agents can work simultaneously.\n"
        "- Merge sub-agent outputs into a cohesive, well-cited final deliverable."
    )
    return Agent(
        name="Overview Manager",
        description="Orchestrates research, citations, and writing to produce the overview.",
        role=role,
        llm=llm,
        tools=list(subagents),
        inference_mode=InferenceMode.FUNCTION_CALLING,
        parallel_tool_calls_enabled=True,
        max_loops=12,
        behaviour_on_max_loops=Behavior.RETURN,
    )


def _build_tools(llm):
    search_tool = ScaleSerpTool(connection=ScaleSerp())
    scraper_tool = ScraperSummarizerTool(connection=ZenRows(), llm=llm)
    return search_tool, scraper_tool


def _build_llm(model_provider: str | None = None, model_name: str | None = None):
    kwargs = {}
    if model_provider:
        kwargs["model_provider"] = model_provider
    if model_name:
        kwargs["model_name"] = model_name
    return setup_llm(**kwargs)


def run_literature_overview(
    topic: str,
    focus: str | None = None,
    model_provider: str | None = None,
    model_name: str | None = None,
) -> dict:
    """Execute the literature overview workflow and return the manager agent output."""
    llm = _build_llm(model_provider=model_provider, model_name=model_name)
    search_tool, scraper_tool = _build_tools(llm)

    researcher = _research_agent(llm, search_tool, scraper_tool)
    citation_curator = _citation_agent(llm)
    writer = _writer_agent(llm)
    manager = _manager_agent(llm, [researcher, citation_curator, writer])

    workflow = Workflow(flow=Flow(nodes=[manager]))

    base_prompt = (
        "Prepare a literature overview on the following topic. Emphasize recent work, "
        "contrasting viewpoints, and practical applications. Close with open research questions.\n\n"
        f"Topic: {topic.strip()}"
    )
    if focus:
        base_prompt += f"\nFocus areas: {focus.strip()}"

    result = workflow.run(input_data={"input": base_prompt})
    logger.info("Literature overview workflow completed")
    return result.output.get(manager.id, {})


def main() -> None:
    topic = input("Enter the topic for the literature overview: ")
    focus = input("Optional focus areas or constraints (press Enter to skip): ")
    focus = focus if focus.strip() else None

    manager_output = run_literature_overview(topic=topic, focus=focus)
    content = manager_output.get("output", {}).get("content")

    if content:
        print(content)
        OUTPUT_FILE_PATH.write_text(content)
        print(f"\nSaved literature overview to {OUTPUT_FILE_PATH.resolve()}")
    else:
        print("Workflow completed without textual content. Raw output:")
        print(manager_output)


if __name__ == "__main__":
    main()
