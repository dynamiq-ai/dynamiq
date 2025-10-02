from dynamiq.memory import Memory
from dynamiq.memory.backends.in_memory import InMemory
from dynamiq.nodes.agents import Agent
from examples.llm_setup import setup_llm


def setup_agent():
    llm = setup_llm()
    memory = Memory(backend=InMemory())
    AGENT_ROLE = "Helpful assistant with the goal of providing useful information and answering questions."
    agent = Agent(
        name="Agent",
        llm=llm,
        role=AGENT_ROLE,
        id="agent",
        memory=memory,
    )
    return agent


def chat_loop(agent):
    print("Welcome to the AI Chat! (Type 'exit' to end)")
    while True:
        user_input = input("You: ")
        user_id = "default"
        session_id = "default"
        if user_input.lower() == "exit":
            break

        response = agent.run({"input": user_input, "user_id": user_id, "session_id": session_id})
        response_content = response.output.get("content")
        print(f"AI: {response_content}")

    print("\nChat History:")
    print(agent.memory.get_all_messages_as_string())


if __name__ == "__main__":
    chat_agent = setup_agent()
    chat_loop(chat_agent)
