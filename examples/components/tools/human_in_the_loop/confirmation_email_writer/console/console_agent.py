from dynamiq.nodes.agents import Agent
from dynamiq.nodes.tools.human_feedback import HumanFeedbackTool
from dynamiq.nodes.tools.python import Python
from dynamiq.nodes.types import InferenceMode
from dynamiq.types.feedback import ApprovalConfig, FeedbackMethod
from examples.llm_setup import setup_llm

PYTHON_TOOL_CODE = """
def run(inputs):
    return {"content": "Email was sent."}
"""


def run_agent(query) -> dict:
    """
    Creates agent

    Returns:
        dict: Agent final output.
    """
    llm = setup_llm()

    email_sender_tool = Python(
        name="EmailSenderTool",
        description="Sends email. Put all email in string under 'email' key. ",
        code=PYTHON_TOOL_CODE,
        approval=ApprovalConfig(
            enabled=True,
            feedback_method=FeedbackMethod.CONSOLE,
            msg_template=(
                "Email sketch: {{input_data.email}}.\n"
                "Approve or cancel email sending. Send nothing for approval;"
                "provide feedback to cancel and regenerate."
            ),
        ),
    )

    human_feedback_tool = HumanFeedbackTool(
        name="human-feedback",
        description="Tool for human interaction. Use action='ask' to request clarifications, action='info' "
        "to notify user.",
        input_method=FeedbackMethod.CONSOLE,
        output_method=FeedbackMethod.CONSOLE,
    )

    agent = Agent(
        name="research_agent",
        role=(
            "You are a helpful assistant that has access to the internet using Tavily Tool."
            "You can request clarifications or send messages using human-feedback tool with "
            "action='ask' or action='info'."
        ),
        inference_mode=InferenceMode.XML,
        llm=llm,
        tools=[email_sender_tool, human_feedback_tool],
    )

    return agent.run(
        input_data={
            "input": f"Write and send email: {query}. Notify user about status using human-feedback tool "
            "with action='info'."
        },
    ).output["content"]


if __name__ == "__main__":
    query = input("Describe details of the email: ")
    result = run_agent(query)
    print("Result: ")
    print(result)
