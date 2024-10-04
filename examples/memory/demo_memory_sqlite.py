from dynamiq.memory.backend import SQLite
from dynamiq.memory.memory import Memory
from dynamiq.prompts import MessageRole

backend = SQLite()
memory = Memory(backend=backend)

memory.add_message(MessageRole.USER, "This is a test message for SQLite.", metadata={"category": "testing"})
memory.add_message(MessageRole.ASSISTANT, "SQLite is working!", metadata={"category": "testing"})

# Search with filters only
results = memory.search_messages(filters={"category": "testing"})
print("SQLite results with filters only:", [r.content for r in results])

# Search with query and filters
results = memory.search_messages(query="SQLite", filters={"category": "testing"})
print("SQLite results with query and filters:", [r.content for r in results])

# Search with query only
results = memory.search_messages("working")
print("SQLite results with query only:", [r.content for r in results])

memory.clear()
print("Is memory empty?", memory.is_memory_empty())
