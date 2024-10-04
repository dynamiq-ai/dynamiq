from dynamiq.components.embedders.openai import OpenAIEmbedder
from dynamiq.connections import Pinecone as PineconeConnection
from dynamiq.memory import Config, Memory
from dynamiq.memory.backend import Pinecone
from dynamiq.prompts import MessageRole

pinecone_connection = PineconeConnection()
embedder = OpenAIEmbedder()
config = Config()
backend = Pinecone(connection=pinecone_connection, embedder=embedder, index_name="your-index-name")
memory = Memory(config=config, backend=backend)

memory.add_message(MessageRole.USER, "What's the weather in London?", metadata={"location": "London"})
memory.add_message(MessageRole.ASSISTANT, "It's sunny in London.", metadata={"location": "London"})
memory.add_message(MessageRole.USER, "What about Paris?", metadata={"location": "Paris"})
memory.add_message(MessageRole.ASSISTANT, "It's raining in Paris.", metadata={"location": "Paris"})


# Search with filters only
results = memory.search_messages(filters={"location": "London"})
print("Pinecone results with filters only:", [r.content for r in results])

# Search with query and filters
results = memory.search_messages(query="weather", filters={"location": "London"})
print("Pinecone results with query and filters:", [r.content for r in results])

# Search with query only
results = memory.search_messages("Paris")
print("Pinecone results with query only:", [r.content for r in results])

memory.clear()
