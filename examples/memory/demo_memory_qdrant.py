from dynamiq.components.embedders.openai import OpenAIEmbedder
from dynamiq.connections import Qdrant as QdrantConnection
from dynamiq.memory.backend.qdrant import Qdrant
from dynamiq.memory.memory import Memory
from dynamiq.prompts import MessageRole

# Initialize Qdrant connection
qdrant_connection = QdrantConnection()

# Initialize embedder
embedder = OpenAIEmbedder()

# Initialize Qdrant backend
qdrant_backend = Qdrant(connection=qdrant_connection, embedder=embedder)

# Initialize Memory with Qdrant backend
memory = Memory(backend=qdrant_backend)

# Add messages
memory.add_message(role=MessageRole.USER, content="Hello, how are you?")
memory.add_message(role=MessageRole.ASSISTANT, content="I'm doing well, thank you!")

# Search for messages
search_results = memory.search_messages(query="how are you")
print(search_results)

# Get all messages
all_messages = memory.get_all_messages()
print(all_messages)

# Clear the memory
memory.clear()
