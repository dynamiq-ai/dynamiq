from typing import Any

import yaml
from states import (
    additional_reflection_review,
    generate_caption,
    generate_citation,
    generate_references,
    internet_search_state,
    reflection_review,
    review_topic_sentence,
    suggest_title_review_state,
    suggest_title_state,
    write_abstract,
    write_paper,
    write_paper_review,
    write_topic,
)

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.agents.orchestrators.graph import END, GraphOrchestrator
from dynamiq.nodes.agents.orchestrators.graph_manager import GraphAgentManager
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.runnables import RunnableResult

STATES = {
    "internet_search": internet_search_state,
    "write_topic": write_topic,
    "review_topic_sentence": review_topic_sentence,
    "write_paper": write_paper,
    "write_paper_review": write_paper_review,
    "reflection_review": reflection_review,
    "additional_reflection_review": additional_reflection_review,
    "write_abstract": write_abstract,
    "generate_references": generate_references,
    "generate_caption": generate_caption,
    "generate_citation": generate_citation,
}


def is_paper_review_complete(context: dict[str, Any]):
    print(context.get("max_revisions"))
    print("Max revisions")
    print(context.get("revision_number"))
    print("revision_number")
    if context.get("update_instruction"):
        return "write_paper_review"
    elif context.get("revision_number") <= context.get("max_revisions"):
        return "reflection_review"
    else:
        return "write_abstract"


def create_orchestrator(configuration) -> GraphOrchestrator:
    """
    Creates orchestrator

    Returns:
        GraphOrchestrator: The configured orchestrator.
    """
    llm = OpenAI(
        name="OpenAI LLM",
        connection=OpenAIConnection(),
        model="gpt-4o-mini",
    )

    suggest_title = configuration.get("suggest_title", False)

    orchestrator = GraphOrchestrator(
        name="Graph orchestrator",
        manager=GraphAgentManager(llm=llm),
        context=configuration,
        initial_state="suggest_title" if suggest_title else "internet_search",
    )

    for name, state in STATES.items():
        orchestrator.add_node(name, [state])

    if suggest_title:
        orchestrator.add_node("suggest_title", [suggest_title_state])
        orchestrator.add_node("suggest_title_review", [suggest_title_review_state])

    orchestrator.add_edge("suggest_title", "suggest_title_review")
    orchestrator.add_edge("suggest_title_review", "internet_search")

    orchestrator.add_edge("internet_search", "write_topic")
    orchestrator.add_edge("write_topic", "review_topic_sentence")
    orchestrator.add_edge("review_topic_sentence", "write_paper")
    orchestrator.add_edge("write_paper", "write_paper_review")

    orchestrator.add_conditional_edge(
        "write_paper_review", ["write_paper_review", "reflection_review", "write_abstract"], is_paper_review_complete
    )

    orchestrator.add_edge("reflection_review", "additional_reflection_review")
    orchestrator.add_edge("additional_reflection_review", "write_paper_review")

    orchestrator.add_edge("write_abstract", "generate_references")
    orchestrator.add_edge("generate_references", "generate_citation")
    orchestrator.add_edge("generate_citation", "generate_caption")

    orchestrator.add_edge("generate_caption", END)

    return orchestrator


def parse_configuration(filename):
    stream = open(filename)
    state = yaml.safe_load(stream)
    return state


def run_orchestrator(orchestrator: GraphOrchestrator, configuration_context: dict) -> RunnableResult:
    """Runs orchestrator"""

    result = orchestrator.run(
        input_data={
            "input": f"Write {configuration_context["type_of_document"]} about {configuration_context["title"]}"
        },
        config=None,
    )

    return result.output["content"]
