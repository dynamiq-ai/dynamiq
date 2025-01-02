from typing import Any

from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.nodes.agents.orchestrators.graph import END, START, GraphOrchestrator
from dynamiq.nodes.agents.orchestrators.graph_manager import GraphAgentManager
from dynamiq.prompts import Message, Prompt
from dynamiq.runnables import RunnableConfig, RunnableResult
from dynamiq.utils.logger import logger
from examples.llm_setup import setup_llm


def create_orchestrator() -> GraphOrchestrator:
    """
    Creates orchestrator

    Returns:
        GraphOrchestrator: The configured orchestrator.
    """
    llm = setup_llm(model_provider="gpt", model_name="gpt-4o-mini", temperature=0)

    def generate_sketch(context: dict[str, Any]):
        "Generate sketch"
        messages = context.get("messages")

        if feedback := context.get("feedback"):
            messages += [Message(role="user", content=f"Generate text again taking into account feedback {feedback}")]

        response = llm.run(
            input_data={},
            prompt=Prompt(
                messages=messages,
            ),
        ).output["content"]

        context["messages"] += [
            Message(
                role="assistant",
                content=f"\n{response}",
            )
        ]

        return {"result": "Generated draft", **context}

    def accept_sketch(context: dict[str, Any]):

        result = input(
            f"Approve whether to publish by providing nothing or cancel by typing in feedback: "
            f"{context["messages"][-1]["content"]}"
        )

        if result:
            return "generate_sketch"

        logger.info("Post was successfully published!")
        return END

    orchestrator = GraphOrchestrator(
        name="Graph orchestrator",
        manager=GraphAgentManager(llm=llm),
    )

    orchestrator.add_state_by_tasks("generate_sketch", [generate_sketch])

    orchestrator.add_edge(START, "generate_sketch")
    orchestrator.add_conditional_edge("generate_sketch", ["generate_sketch", END], accept_sketch)

    return orchestrator


def run_orchestrator(orchestrator, request="Write and publish small post about AI in Sales.") -> RunnableResult:
    """Runs orchestrator"""
    orchestrator = create_orchestrator()
    orchestrator.context = {
        "messages": [Message(role="user", content=f"Return nothing but text: {request}")],
    }
    tracing = TracingCallbackHandler()

    _ = orchestrator.run(
        input_data={"input": request},
        config=RunnableConfig(callbacks=[tracing]),
    )

    return orchestrator.context.get("messages")[-1].content


if __name__ == "__main__":
    result = run_orchestrator("Write and publish small post about AI in Sales.")
    print("Result:")
    print(result)
