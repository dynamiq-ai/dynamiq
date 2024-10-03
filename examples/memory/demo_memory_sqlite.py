from dynamiq.memory.backend import SQLite
from dynamiq.memory.memory import Memory
from dynamiq.prompts import MessageRole

backend = SQLite()
memory_sqlite = Memory(backend=backend)

# Add messages
memory_sqlite.add_message(MessageRole.USER, "This is a message for SQLite.")
memory_sqlite.add_message(MessageRole.ASSISTANT, "SQLite is a great embedded database.")

# Get all messages
print("\nSQLite messages:")
messages = memory_sqlite.get_messages()
for msg in messages:
    print(f"{msg.role}: {msg.content}")

# Search for messages
print("\nSQLite search results:")
search_results = memory_sqlite.search_messages("SQLite")
for msg in search_results:
    print(f"Search result: {msg.content}")

# Print all messages
print("\nSQLite messages:")
print(memory_sqlite.get_messages_as_string())

# Clear memory
memory_sqlite.clear()
print("\nIs SQLite memory empty?", memory_sqlite.is_memory_empty())
