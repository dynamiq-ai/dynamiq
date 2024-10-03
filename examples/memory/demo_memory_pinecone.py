from dynamiq.components.embedders.openai import OpenAIEmbedder
from dynamiq.connections import Pinecone as PineconeConnection
from dynamiq.memory import Config, Memory
from dynamiq.memory.backend import Pinecone
from dynamiq.prompts import MessageRole

pinecone_connection = PineconeConnection()
embedder = OpenAIEmbedder()
config = Config()
backend = Pinecone(connection=pinecone_connection, embedder=embedder)

memory = Memory(
    config=config,
    backend=backend,
)

# Add messages
memory.add_message(MessageRole.USER, "What's the capital of France?")
memory.add_message(MessageRole.ASSISTANT, "The capital of France is Paris.")
memory.add_message(MessageRole.USER, "Can you tell me about the Eiffel Tower?")
memory.add_message(
    MessageRole.ASSISTANT,
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris. It's named after engineer Gustave Eiffel and was constructed from 1887 to 1889 as the entrance arch to the 1889 World's Fair.",  # noqa E501
)

# Get all messages
print("All messages:")
messages = memory.get_all_messages()
for msg in messages:
    print(f"{msg.role}: {msg.content}")

print("\n")

# Search for messages
print("Search results for 'Paris':")
search_results = memory.search_messages("Paris")
for msg in search_results:
    print(f"Search result: {msg.content}")

print("\n")

# Add more messages
memory.add_message(MessageRole.USER, "What's the population of Tokyo?")
memory.add_message(
    MessageRole.ASSISTANT,
    "As of 2021, the estimated population of Tokyo is approximately 14 million for the city proper, and around 37 million for the greater Tokyo metropolitan area.",  # noqa E501
)

# Search again
print("Search results for 'population':")
search_results = memory.search_messages("population")
for msg in search_results:
    print(f"Search result: {msg.content}")

print("\n")

# Check if memory is empty
print("Is memory empty?", memory.is_memory_empty())

# Clear memory
memory.clear()
print("Memory cleared.")
print("Is memory empty now?", memory.is_memory_empty())
