from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.connections import Qdrant as QdrantConnection
from dynamiq.memory.backends.qdrant import Qdrant
from dynamiq.memory.memory import Memory
from dynamiq.nodes.embedders import OpenAIDocumentEmbedder
from dynamiq.prompts import MessageRole

INDEX_NAME = "conversations"
qdrant_connection = QdrantConnection(
    api_key="VpZFKICUAqWc8QrM9bjUhZ_1Rcowz68p_4P-QhcIX7Ua1_5hRrTkoA",
    url="https://b5933e17-450f-410c-b82f-1f8facb74b83.eu-central-1-0.aws.cloud.qdrant.io:6333",
)
openai_connection = OpenAIConnection()
embedder = OpenAIDocumentEmbedder(connection=openai_connection)

qdrant_backend = Qdrant(
    connection=qdrant_connection, embedder=embedder, index_name=INDEX_NAME, create_if_not_exist=True
)
memory = Memory(backend=qdrant_backend)

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
