from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.connections import Pinecone as PineconeConnection
from dynamiq.memory import Memory
from dynamiq.memory.backends import Pinecone
from dynamiq.nodes.embedders import OpenAIDocumentEmbedder
from dynamiq.prompts import MessageRole
from dynamiq.storages.vector.pinecone.pinecone import PineconeIndexType

pinecone_connection = PineconeConnection()
openai_connection = OpenAIConnection()
embedder = OpenAIDocumentEmbedder(connection=openai_connection)

backend = Pinecone(
    index_name="test-conv",
    connection=pinecone_connection,
    embedder=embedder,
    index_type=PineconeIndexType.SERVERLESS,
    cloud="aws",
    region="us-east-1",
)

memory = Memory(backend=backend)


# Add messages with metadata
memory.add(MessageRole.USER, "My favorite color is blue.", metadata={"topic": "colors", "user_id": "123"})
memory.add(MessageRole.ASSISTANT, "Blue is a calming color.", metadata={"topic": "colors", "user_id": "123"})
memory.add(MessageRole.USER, "I like red too.", metadata={"topic": "colors", "user_id": "456"})
memory.add(MessageRole.ASSISTANT, "Red is a passionate color.", metadata={"topic": "colors", "user_id": "456"})


# Get all messages
messages = memory.get_all()
print("All messages:")
for msg in messages:
    print(f"{msg.role}: {msg.content}")


# Search with query only
results = memory.search("red")
print("Results with query only:", [r.content for r in results])
# Search with query and filters
results = memory.search(query="color", filters={"user_id": "123"})
print("Results with query and filter:", [r.content for r in results])
# Search with filters only
results = memory.search(filters={"user_id": "123"})
print("Results with filter only:", [r.content for r in results])

# Clear memory
memory.clear()
print("Is memory empty?", memory.is_empty())
