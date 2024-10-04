# Example usage with InMemory backend and filters:
from dynamiq.memory import Memory
from dynamiq.memory.backend import InMemory
from dynamiq.prompts import MessageRole

# Create a memory instance with InMemory backend
memory = Memory(backend=InMemory())

# Add messages with metadata
memory.add_message(MessageRole.USER, "My favorite color is blue.", metadata={"topic": "colors", "user_id": "123"})
memory.add_message(MessageRole.ASSISTANT, "Blue is a calming color.", metadata={"topic": "colors", "user_id": "123"})
memory.add_message(MessageRole.USER, "I like red too.", metadata={"topic": "colors", "user_id": "456"})

# Search with filters only
results = memory.search_messages(filters={"user_id": "123"})
print("InMemory Results with filter only:", [r.content for r in results])

# Search with query and filters
results = memory.search_messages(query="color", filters={"user_id": "123"})
print("InMemory Results with query and filter:", [r.content for r in results])

# Search with query only
results = memory.search_messages("red")
print("InMemory Results with query only:", [r.content for r in results])

# Get all messages
messages = memory.get_all_messages()
for msg in messages:
    print(f"{msg.role}: {msg.content}")

# Search for messages
search_results = memory.search_messages("math problem")
for msg in search_results:
    print(f"Search result: {msg.content}")

# Clear memory
memory.clear()
print("Is memory empty?", memory.is_memory_empty())
