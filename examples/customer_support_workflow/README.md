# Customer Support Workflow

This project demonstrates a customer support workflow using AI agents and simulated banking tools. It showcases the use of Retrieval-Augmented Generation (RAG) and API simulation for handling customer inquiries in a banking context.

## Project Structure

```
customer_support_workflow/
    bank_rag_tool.py
    __init__.py
    bank_api_sim.py
    main.py
```

## Components

### 1. Bank RAG Tool (bank_rag_tool.py)

This module implements a Retrieval-Augmented Generation (RAG) tool for accessing and querying internal bank documents and policies. It uses the following components:

- PineconeVectorStore for document storage
- OpenAITextEmbedder for text embedding
- PineconeDocumentRetriever for document retrieval
- OpenAI GPT-4 for answer generation

The BankRagSim class provides a method to query the bank's internal documents and generate responses based on the retrieved information.

### 2. Bank API Simulator (bank_api_sim.py)

This module simulates an internal bank API. It uses OpenAI's GPT-4 model to generate responses that mimic a real bank system API. The BankApiSim class can handle various types of requests related to clients, customers, transactions, and accounts.

### 3. Main Workflow (main.py)

The main script orchestrates the customer support workflow using the following components:

- ManagerAgent: Oversees the workflow and delegates tasks to specialized agents
- ReActAgent: Implements agents for bank support and documentation queries
- LinearOrchestrator: Manages the workflow execution
- OpenAI and Anthropic language models

## Setup and Usage

1. Install the required dependencies (dynamiq and its components).
2. Set up API keys for OpenAI and Anthropic as environment variables.
3. Configure Pinecone for document storage (for the RAG tool).
4. Run the main script: `python customer_support_workflow/main.py`

## Workflow Overview

1. The main script initializes two specialized agents:
   - Bank Support Agent: Handles queries using the simulated bank API
   - Bank Documentation Agent: Handles queries using the RAG tool for internal documents
2. A ManagerAgent oversees the workflow and delegates tasks to the appropriate agent.
3. The LinearOrchestrator manages the execution of the workflow, passing the customer query through the agents.
4. The system generates a response based on the combined knowledge from the bank API and internal documentation.

## Customization

You can modify the `main.py` script to change the language model provider (OpenAI or Anthropic) and adjust the agent configurations as needed.

## Note

This project is for demonstration purposes and showcases how AI agents can be used to create a customer support workflow in a banking context. The bank API and document retrieval are simulated, but the structure can be adapted for use with real banking systems and document repositories.
