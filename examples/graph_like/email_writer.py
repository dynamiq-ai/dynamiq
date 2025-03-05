from typing import Any

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes import InputTransformer
from dynamiq.nodes.agents.orchestrators.graph import END, START, GraphOrchestrator
from dynamiq.nodes.agents.orchestrators.graph_manager import GraphAgentManager
from dynamiq.nodes.agents.simple import SimpleAgent
from dynamiq.nodes.llms import OpenAI

llm = OpenAI(
    connection=OpenAIConnection(api_key="OPENAI_API_KEY"),
    model="gpt-4o",
    temperature=0.1,
)

email_writer = SimpleAgent(
    name="email-writer-agent",
    llm=llm,
    role="Write personalized emails taking into account feedback.",
    input_transformer=InputTransformer(
        selector={
            "input": "$.context.agent_input",
        },
    ),
)


def gather_feedback(context: dict[str, Any], **kwargs):
    """Gather feedback about email draft."""
    draft = context.get("history", [{}])[-1].get("content", "No draft")

    feedback = input(
        f"Email draft:\n"
        f"{draft}\n"
        f"Type in SEND to send email, CANCEL to exit, or provide feedback to refine email: \n"
    )

    result = f"Gathered feedback: {feedback}"

    feedback = feedback.strip().lower()
    if feedback == "":
        print("####### Email was sent! #######")
        result = "Email was sent!"
        agent_input = None
    else:
        print("####### Email was canceled! #######")
        result = "Email was canceled!"
        agent_input = f"Draft of canceled email: \n{draft}\n" f"Feedback of user about this draft: \n{feedback}"

    return {"result": result, "agent_input": agent_input}


def router(context: dict[str, Any], **kwargs):
    """Determines next state based on provided feedback."""
    if context.get("agent_input"):
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
    orchestrator.run(
        input_data={"input": f"Write and post email, provide feedback about status of email: {email_details}"}
    )
    print(orchestrator._chat_history[-1]["content"])
