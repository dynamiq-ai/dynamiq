from dynamiq.memory.memory import Memory
from dynamiq.prompts import MessageRole

# Create a memory instance with default in-memory storage
memory = Memory()

# Add messages
memory.add_message(MessageRole.USER, "Hello, how are you?")
memory.add_message(MessageRole.ASSISTANT, "I'm doing well, thank you for asking. How can I assist you today?")
memory.add_message(MessageRole.USER, "Can you help me with a math problem?")

# Get all messages
messages = memory.get_messages()
for msg in messages:
    print(f"{msg.role}: {msg.content}")

# Search for messages
search_results = memory.search_messages("math problem")
for msg in search_results:
    print(f"Search result: {msg.content}")

# Clear memory
memory.clear()
print("Is memory empty?", memory.is_memory_empty())
