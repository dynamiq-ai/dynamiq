from dynamiq.components.embedders.openai import OpenAIEmbedder
from dynamiq.connections import Pinecone
from dynamiq.memory import Config, Memory
from dynamiq.memory.backend import PineconeBackend
from dynamiq.nodes.agents.simple import SimpleAgent
from examples.llm_setup import setup_llm


def setup_agent():
    llm = setup_llm()

    # Set up Pinecone connection
    pinecone_api_key = "10e1dc8b-7ec1-4d23-8174-fa1f9a23d7aa"
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY environment variable is not set")

    pinecone_connection = Pinecone(api_key=pinecone_api_key)
    embedder = OpenAIEmbedder()

    # Create a memory instance with Pinecone storage
    backend = PineconeBackend(connection=pinecone_connection, index_name="my-memory-index-v2", embedder=embedder)
    config = Config()

    memory_pinecone = Memory(config=config, backend=backend)

    AGENT_ROLE = "helpful assistant"
    AGENT_GOAL = "is to provide useful information and answer questions"
    agent = SimpleAgent(
        name="Agent",
        llm=llm,
        role=AGENT_ROLE,
        goal=AGENT_GOAL,
        id="agent",
        memory=memory_pinecone,
    )
    return agent


def chat_loop(agent):
    print("Welcome to the AI Chat! (Type 'exit' to end)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        # The agent uses the memory internally when generating a response
        response = agent.run({"input": user_input})
        response_content = response.output.get("content")
        print(f"AI: {response_content}")

    print("\nChat History:")
    print(agent.memory.get_messages_as_string())


if __name__ == "__main__":
    chat_agent = setup_agent()
    chat_loop(chat_agent)
