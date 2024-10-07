from dynamiq.components.embedders.openai import OpenAIEmbedder
from dynamiq.connections import Qdrant as QdrantConnection
from dynamiq.memory import Config, Memory
from dynamiq.memory.backend import Qdrant
from dynamiq.nodes.agents.simple import SimpleAgent
from dynamiq.prompts import MessageRole
from examples.llm_setup import setup_llm

USER_ID = "01"
MEMORY_NAME = "user-01"
AGENT_ROLE = "friendly helpful assistant"


def setup_agent():
    llm = setup_llm(model_provider="gpt", model_name="gpt-4o-mini", temperature=0.5)
    qdrant_connection = QdrantConnection()
    embedder = OpenAIEmbedder()

    # Create a memory instance with Qdrant storage
    backend = Qdrant(connection=qdrant_connection, embedder=embedder, collection_name=MEMORY_NAME)
    config = Config()

    memory = Memory(config=config, backend=backend)
    memory.add_message(
        MessageRole.USER, "Hey! I'm Oleksii, machine learning engineer from Dynamiq.", metadata={"user_id": USER_ID}
    )
    memory.add_message(
        MessageRole.USER,
        "My hobbies are: tennis, reading and cinema. I prefer science and sci-fi books.",
        metadata={"user_id": USER_ID},
    )

    agent = SimpleAgent(
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
        if user_input.lower() == "exit":
            break

        # The agent uses the memory internally when generating a response
        response = agent.run({"input": user_input, "user_id": USER_ID})
        response_content = response.output.get("content")
        print(f"AI: {response_content}")

    print("\nChat History:")
    print(agent.memory.get_all_messages_as_string())


if __name__ == "__main__":
    chat_agent = setup_agent()
    chat_loop(chat_agent)
