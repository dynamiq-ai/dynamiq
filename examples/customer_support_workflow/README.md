# Customer Support Workflow Example

This directory contains an example of a customer support workflow built using Dynamiq agents and tools. The workflow demonstrates how to orchestrate multiple agents to handle different types of customer requests, such as accessing internal bank APIs and retrieving information from documentation.

## Components

### `bank_api_sim.py`

- Simulates an internal bank API using OpenAI's language model.
- Responds to queries about account information, transactions, etc., in JSON format.

### `bank_rag_tool.py`

- Implements a Retrieval-Augmented Generation (RAG) tool for accessing bank documentation.
- Uses Pinecone for document retrieval and OpenAI for text embedding and answer generation.

### `main.py`

- Defines the main workflow logic.
- Creates instances of `ReActAgent` for handling API and documentation queries.
- Creates a `LinearOrchestrator` to manage the workflow of multiple agents.
- Executes the workflow with a sample input.

## Workflow Logic

1. The user provides a query (e.g., "fast block my card").
2. The `LinearOrchestrator` receives the query and determines which agent should handle it.
3. If the query relates to internal bank APIs, the `agent_bank_support` (using `BankApiSim`) is invoked.
4. If the query relates to bank documentation, the `agent_bank_documentation` (using `BankRAGTool`) is invoked.
5. The selected agent processes the query using its tools and returns a response.
6. The `LinearOrchestrator` receives the agent's response and returns it to the user.

## Usage

1. **Set up environment variables:**
   - `OPENAI_API_KEY`: Your OpenAI API key.
   - `OPENAI_MODEL`: The OpenAI model to use (defaults to "gpt-4").
   - `PINECONE_API_KEY`: Your Pinecone API key.
   - `PINECONE_ENVIRONMENT`: Your Pinecone environment.
2. **Run the workflow:** `python main.py`

## Key Concepts

- **Agent Orchestration:** Managing the interaction and collaboration of multiple agents to solve complex tasks.
- **Retrieval-Augmented Generation (RAG):** Combining information retrieval with language model generation to provide more accurate and comprehensive answers.
- **Tool Usage:** Leveraging specialized tools to extend the capabilities of agents.
- **Human Feedback:** Integrating human feedback to improve the accuracy and reliability of agents.
