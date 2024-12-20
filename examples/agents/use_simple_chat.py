from dynamiq.nodes.agents.simple import SimpleAgent
from dynamiq.prompts import Message, MessageRole
from examples.llm_setup import setup_llm


def setup_agent():
    llm = setup_llm()
    AGENT_ROLE = "Helpful assistant with the goal of providing useful information and answering questions."
    agent = SimpleAgent(
        name="Agent",
        llm=llm,
        role=AGENT_ROLE,
        id="agent",
    )
    return agent


def chat_loop(agent):
    messages = []
    print("Welcome to the AI Chat! (Type 'exit' to end)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        messages.append(Message(role=MessageRole.USER, content=user_input))
        response = agent.run({"input": user_input, "chat_history": messages})
        response_content = response.output.get("content")
        messages.append(Message(role=MessageRole.ASSISTANT, content=response_content))
        print(f"AI: {response_content}")


if __name__ == "__main__":
    chat_agent = setup_agent()
    chat_loop(chat_agent)
