from dynamiq.nodes.agents import ReActAgent
from dynamiq.nodes.tools.human_feedback import HumanFeedbackTool
from dynamiq.nodes.tools.python import Python
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
        description="This tool can be used to request some clarifications from user.",
        input_method=FeedbackMethod.CONSOLE,
    )

    agent = ReActAgent(
        name="research_agent",
        role="You are a helpful assistant that has access to the internet using Tavily Tool. ",
        llm=llm,
        tools=[email_sender_tool, human_feedback_tool],
    )

    return agent.run(
        input_data={"input": f"Write and send email. {query}"},
    ).output["content"]


if __name__ == "__main__":
    query = input("Describe details of the email: ")
    result = run_agent(query)
    print("Result: ")
    print(result)
