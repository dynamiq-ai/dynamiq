from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.connections import Qdrant as QdrantConnection
from dynamiq.memory import Memory
from dynamiq.memory.backends import Qdrant
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.embedders import OpenAIDocumentEmbedder
from examples.llm_setup import setup_llm


def setup_agent():
    llm = setup_llm()
    qdrant_connection = QdrantConnection()
    openai_connection = OpenAIConnection()
    embedder = OpenAIDocumentEmbedder(connection=openai_connection)

    backend = Qdrant(
        connection=qdrant_connection,
        embedder=embedder,
        index_name="default",
    )

    memory_qdrant = Memory(backend=backend)

    AGENT_ROLE = "Helpful assistant with the goal of providing useful information and answering questions."
    agent = Agent(
        name="Agent",
        llm=llm,
        role=AGENT_ROLE,
        id="agent",
        memory=memory_qdrant,
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

        response = agent.run(
            {
                "input": user_input,
                "user_id": user_id,
                "session_id": session_id,
            }
        )
        response_content = response.output.get("content")
        print(f"AI: {response_content}")

    print("\nChat History:")
    print(agent.memory.get_all_messages_as_string())


if __name__ == "__main__":
    chat_agent = setup_agent()
    chat_loop(chat_agent)
