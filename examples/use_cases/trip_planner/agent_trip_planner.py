"""Trip planner example using a manager agent with specialized sub-agents as tools."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from dynamiq import Workflow
from dynamiq.connections import ScaleSerp
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.tools.scale_serp import ScaleSerpTool
from dynamiq.nodes.types import Behavior, InferenceMode
from dynamiq.utils.logger import logger

from examples.llm_setup import setup_llm
from examples.use_cases.trip_planner.prompts import generate_customer_prompt

OUTPUT_FILE_PATH = Path("trip_plan.md")


def _city_selection_agent(llm, search_tool: ScaleSerpTool) -> Agent:
    role = (
        "You are the City Selection sub-agent.\n"
        "- Always expect tool input as {'input': '<travel task>'}.\n"
        "- Compare candidate destinations using the ScaleSerp search tool.\n"
        "- Justify your recommendation with weather, events, and cost insights."
    )
    return Agent(
        name="City Selection Expert",
        description="Evaluates candidate cities using web search data.",
        role=role,
        llm=llm,
        tools=[search_tool],
        inference_mode=InferenceMode.FUNCTION_CALLING,
        max_loops=8,
        behaviour_on_max_loops=Behavior.RETURN,
    )


def _city_guide_agent(llm, search_tool: ScaleSerpTool) -> Agent:
    role = (
        "You are the City Guide sub-agent.\n"
        "- Always expect tool input as {'input': '<city to research>'}.\n"
        "- Enrich the guide with logistics, local tips, and daily highlights.\n"
        "- Use ScaleSerp search results for up-to-date details when needed."
    )
    return Agent(
        name="City Guide Expert",
        description="Builds a rich guide for the selected destination.",
        role=role,
        llm=llm,
        tools=[search_tool],
        inference_mode=InferenceMode.FUNCTION_CALLING,
        max_loops=8,
        behaviour_on_max_loops=Behavior.RETURN,
    )


def _itinerary_agent(llm) -> Agent:
    role = (
        "You are the Itinerary Writer sub-agent.\n"
        "- Always expect tool input as {'input': '<city insights>'}.\n"
        "- Produce a structured 7-day itinerary with budgets and packing tips.\n"
        "- Combine research provided by the other agents without inventing data."
    )
    return Agent(
        name="Itinerary Writer",
        description="Transforms research into a polished multi-day travel plan.",
        role=role,
        llm=llm,
        inference_mode=InferenceMode.FUNCTION_CALLING,
        max_loops=6,
        behaviour_on_max_loops=Behavior.RETURN,
    )


def _manager_agent(llm, subagents: Iterable[Agent]) -> Agent:
    role = (
        "You are the Trip Planner Manager coordinating specialist agents.\n"
        "- Decompose the travel request and delegate tasks strategically.\n"
        "- When invoking a sub-agent tool ALWAYS pass {'input': '<task>'}.\n"
        "- Prefer parallel calls when research can happen independently.\n"
        "- Synthesize the sub-agent outputs into a final actionable itinerary."
    )
    return Agent(
        name="Trip Planner Manager",
        description="Coordinates sub-agents to research and assemble the travel plan.",
        role=role,
        llm=llm,
        tools=list(subagents),
        inference_mode=InferenceMode.FUNCTION_CALLING,
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


def run_trip_planner(
    input_data: dict,
    model_provider: str | None = None,
    model_name: str | None = None,
) -> dict:
    """Execute the trip planning workflow and return the manager agent output."""
    llm = _build_llm(model_provider=model_provider, model_name=model_name)
    search_tool = ScaleSerpTool(connection=ScaleSerp())

    city_selector = _city_selection_agent(llm, search_tool)
    city_guide = _city_guide_agent(llm, search_tool)
    itinerary_writer = _itinerary_agent(llm)
    manager = _manager_agent(llm, [city_selector, city_guide, itinerary_writer])

    workflow = Workflow(flow=Flow(nodes=[manager]))

    user_prompt = generate_customer_prompt(input_data)
    result = workflow.run(input_data={"input": user_prompt})
    logger.info("Trip planning workflow completed")
    return result.output.get(manager.id, {})


def main() -> None:
    location = input("Enter your starting location: ")
    cities = input("Enter candidate cities to evaluate (comma separated): ")
    dates = input("Enter the trip dates: ")
    interests = input("Enter traveler interests: ")

    payload = {
        "location": location,
        "cities": cities,
        "dates": dates,
        "interests": interests,
    }

    manager_output = run_trip_planner(payload)
    content = manager_output.get("output", {}).get("content")

    if content:
        print(content)
        OUTPUT_FILE_PATH.write_text(content)
        print(f"\nSaved trip plan to {OUTPUT_FILE_PATH.resolve()}")
    else:
        print("Workflow completed without textual content. Raw output:")
        print(manager_output)


if __name__ == "__main__":
    main()
