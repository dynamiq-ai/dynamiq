from dynamiq.connections import Exa
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.connections import Pinecone as PineconeConnection
from dynamiq.memory import Memory
from dynamiq.memory.backends import Pinecone
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.embedders import OpenAIDocumentEmbedder
from dynamiq.nodes.tools.exa_search import ExaTool
from dynamiq.storages.vector.pinecone.pinecone import PineconeIndexType
from examples.llm_setup import setup_llm


def setup_agent():
    connection_exa = Exa()
    tool_search = ExaTool(connection=connection_exa)

    llm = setup_llm()
    pinecone_connection = PineconeConnection()
    openai_connection = OpenAIConnection()
    embedder = OpenAIDocumentEmbedder(connection=openai_connection)

    backend = Pinecone(
        index_name="oleks",
        connection=pinecone_connection,
        embedder=embedder,
        index_type=PineconeIndexType.SERVERLESS,
        cloud="aws",
        region="us-east-1",
    )

    memory_pinecone = Memory(backend=backend)

    AGENT_ROLE = "Helpful assistant with the goal of providing useful information and answering questions."
    agent = Agent(
        name="Agent",
        llm=llm,
        role=AGENT_ROLE,
        id="agent",
        tools=[tool_search],
        memory=memory_pinecone,
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
