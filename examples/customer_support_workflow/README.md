# Customer Support Workflow Example

This directory contains an example of a bank customer support workflow built using Dynamiq agents and tools. The workflow demonstrates how to integrate LLM Agents with RAG to handle different types of customer requests by accessing internal bank API and its documentation.

## Components

### `bank_api.py`

- Simple API with single endpoint
- Responds to queries in JSON format.

### `bank_rag_tool.py`

- Implements a Retrieval-Augmented Generation (RAG) tool for accessing bank documentation.
- Uses Pinecone for document retrieval and OpenAI for text embedding and answer generation.

### `main.py`
- Defines the main workflow logic.
- Creates workflow with instances of `ReActAgent` for handling API and documentation queries.
- Executes the workflow with a sample input.

## Workflow Logic

1. The user provides a query (e.g., "fast block my card").
2. `RAG Agent` is invoked to find relevant documentation on how to proceed with request.
3. `API Agent` starts with documentation provided by `RAG Agent`. It will gather required informatiom from user and execute operation with API.
4. Upon completion of the operation, a concise summary of the request and its status will be provided.

## Usage

1. **Set up environment variables:**
   - `OPENAI_API_KEY`: Your OpenAI API key.
   - `PINECONE_API_KEY`: Your Pinecone API key.
   - `PINECONE_ENVIRONMENT`: Your Pinecone environment.

2. **Run the workflow:** `python main.py`

## Key Concepts

- **Workflows:** Creating flow of multiple agents to solve complex tasks.
- **Retrieval-Augmented Generation (RAG):** Combining information retrieval with language model generation to provide more accurate and comprehensive answers.
- **Tool Usage:** Leveraging specialized tools to extend the capabilities of agents.
- **Human Feedback:** Integrating human feedback to improve the accuracy and reliability of agents.
