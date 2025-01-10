from typing import Any

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.agents.orchestrators.graph import END, START, GraphOrchestrator
from dynamiq.nodes.agents.orchestrators.graph_manager import GraphAgentManager
from dynamiq.nodes.agents.simple import SimpleAgent
from dynamiq.nodes.llms import OpenAI

llm = OpenAI(
    connection=OpenAIConnection(),
    model="gpt-4o",
    temperature=0.1,
)

email_writer = SimpleAgent(
    name="email-writer-agent",
    llm=llm,
    role="Write personalized emails taking into account feedback. ",
)


def gather_feedback(context: dict[str, Any]):
    """Gather feedback about email draft."""
    feedback = input(
        f"Email draft:\n"
        f"{context["history"][-1]["content"]}\n"
        f"Type in SEND to send email, CANCEL to exit, or provide feedback to refine email: \n"
    )

    reiterate = True

    result = f"Gathered feedback {feedback}"
    if feedback == "SEND":
        print("####### Email was sent! #######")
        result = "Email was sent!"
        reiterate = False
    elif feedback == "CANCEL":
        print("####### Email was canceled! #######")
        result = "Email was canceled!"
        reiterate = False

    return {"result": result, "reiterate": reiterate}


def router(context: dict[str, Any]):
    """Determines next state based on provided feedback."""
    if context.get("reiterate", False):
        return "generate_sketch"

    return END


orchestrator = GraphOrchestrator(
    name="Graph orchestrator",
    manager=GraphAgentManager(llm=llm),
)

orchestrator.add_state_by_tasks("generate_sketch", [email_writer])
orchestrator.add_state_by_tasks("gather_feedback", [gather_feedback])

orchestrator.add_edge(START, "generate_sketch")
orchestrator.add_edge("generate_sketch", "gather_feedback")

orchestrator.add_conditional_edge("gather_feedback", ["generate_sketch", END], router)


if __name__ == "__main__":
    print("Welcome to email writer.")
    email_details = input("Provide email details: ")
    orchestrator.run(input_data={"input": f"Write and post email: {email_details}"})
