from dynamiq.components.embedders.openai import OpenAIEmbedder
from dynamiq.connections import Qdrant as QdrantConnection
from dynamiq.memory.backend.qdrant import Qdrant
from dynamiq.memory.memory import Memory
from dynamiq.prompts import MessageRole

qdrant_connection = QdrantConnection()
embedder = OpenAIEmbedder()
qdrant_backend = Qdrant(connection=qdrant_connection, embedder=embedder, collection_name="your-collection")
memory = Memory(backend=qdrant_backend)


memory.add_message(MessageRole.USER, "Hello, how are you?", metadata={"mood": "happy"})
memory.add_message(MessageRole.ASSISTANT, "I'm doing well, thank you!", metadata={"mood": "helpful"})


# Search with filters only
results = memory.search_messages(filters={"mood": "happy"})
print("Qdrant results with filters only:", [r.content for r in results])

# Search with query and filters
results = memory.search_messages(query="how are you", filters={"mood": "happy"})
print("Qdrant results with query and filters:", [r.content for r in results])

# Search with query only
results = memory.search_messages("thank you")
print("Qdrant results with query only:", [r.content for r in results])

memory.clear()
