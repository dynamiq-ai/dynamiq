# Memory Module Examples

This directory contains examples demonstrating the usage of the `Memory` module within the Dynamiq framework. The `Memory` module provides a flexible and extensible way to manage conversation history and other relevant information for agents, enabling them to maintain context and improve response quality.

## Examples

### Basic In-Memory Storage

- **`demo_memory.py`**: Demonstrates basic memory operations (add, get, search, clear) using the default in-memory storage.
- **`demo_simple_agent_chat_memory.py`**: Showcases the integration of the `Memory` module with a `SimpleAgent` in a chat loop, utilizing in-memory storage.

### Persistent Storage with Pinecone

- **`demo_memory_pinecone.py`**: Illustrates how to use Pinecone as a persistent storage backend for the `Memory` module, including embedding and search functionalities.
- **`demo_simple_agent_chat_memory_pinecone.py`**: Integrates Pinecone-backed memory with a `SimpleAgent` in a chat loop, demonstrating persistent conversation history.

### Persistent Storage with SQLite

- **`demo_memory_sqlite.py`**: Demonstrates the usage of SQLite as a persistent storage backend for the `Memory` module.

### Persistent Storage with Qdrant

- **`demo_memory_qdrant.py`**: Demonstrates the usage of Qdrant as a persistent storage backend for the `Memory` module.


## Usage

Each example file can be run independently to observe the functionality of the `Memory` module in different scenarios.

**Note:** For examples using Pinecone, ensure you have a Pinecone account and set the `PINECONE_API_KEY`, `PINECONE_CLOUD` and `PINECONE_REGION` environment variables. For SQLite examples, a database file (`conversations.db`) will be created in the same directory.

## Key Concepts

- **`Memory`**: The core class for managing conversation history and other data.
- **`Config`**: Configuration object for customizing memory behavior (e.g., storage provider).
- **`Backend`**: Abstract class defining the interface for different storage backends (e.g., InMemory, Pinecone, SQLite).
- **`Embedder`**: Component for generating embeddings of messages for semantic search (used with Pinecone).
