from typing import Any

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.agents.orchestrators.graph import END, START, GraphOrchestrator
from dynamiq.nodes.agents.orchestrators.graph_manager import GraphAgentManager
from dynamiq.nodes.llms import OpenAI
from dynamiq.prompts import Message, Prompt

llm = OpenAI(
    connection=OpenAIConnection(api_key="$OPENAI_API_KEY"),
    model="gpt-4o",
    temperature=0.1,
)


def generate_sketch(context: dict[str, Any]):
    """Generate draft of email"""
    messages = context.get("messages")

    if feedback := context.get("feedback"):
        messages.append(Message(role="user", content=f"Generate text again taking into account feedback {feedback}"))

    response = llm.run(
        input_data={},
        prompt=Prompt(
            messages=messages,
        ),
    ).output["content"]

    messages.append(Message(role="assistant", content=response))

    return {"result": response, "messages": messages}


def gather_feedback(context: dict[str, Any]):
    """Gather feedback about email draft."""
    feedback = input(
        f"Email draft:\n"
        f"{context["messages"][-1]["content"]}\n"
        f"Type in SEND to send email, CANCEL to exit, or provide feedback to refine email: \n"
    )

    return {"result": feedback, "feedback": feedback}


def router(context: dict[str, Any]):
    """Determines next state based on provided feedback."""
    feedback = context.get("feedback")

    if feedback == "SEND":
        print("######### Email was sent! #########")
        return END

    if feedback == "CANCEL":
        print("######### Email was NOT sent! #########")
        return END

    return "generate_sketch"


orchestrator = GraphOrchestrator(
    name="Graph orchestrator",
    manager=GraphAgentManager(llm=llm),
)

orchestrator.add_state_by_tasks("generate_sketch", [generate_sketch])
orchestrator.add_state_by_tasks("gather_feedback", [gather_feedback])

orchestrator.add_edge(START, "generate_sketch")
orchestrator.add_edge("generate_sketch", "gather_feedback")

orchestrator.add_conditional_edge("gather_feedback", ["generate_sketch", END], router)


if __name__ == "__main__":
    print("Welcome to email writer.")
    email_details = input("Provide email details: ")

    orchestrator.context = {
        "messages": [Message(role="user", content=email_details)],
    }

    orchestrator.run(input_data={})