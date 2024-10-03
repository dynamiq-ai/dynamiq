# Agent Workflow Examples

This directory contains examples demonstrating how to use different types of agents within Dynamiq workflows. Agents are autonomous entities that can interact with their environment (e.g., using tools, accessing memory) to achieve goals.

## Examples

### Simple Agent

- **`use_simple_wf.py`**: Demonstrates the usage of a `SimpleAgent` to answer a basic question. The agent utilizes an LLM to generate a response based on its role and goal. It also showcases how to customize the agent's behavior by adding custom blocks of instructions.

### ReAct Agent

- **`use_react_wf.py`**: Showcases the `ReActAgent`, which combines reasoning and action to solve more complex tasks. This example demonstrates how to equip the agent with various tools, including:
    - **`ScaleSerpTool`**: For web search.
    - **`E2BInterpreterTool`**: For executing code.
    - **`Python`**: For custom Python code execution.
    - **`HttpApiCall`**: For interacting with external APIs.
    - **`RetrievalTool`**: For retrieving information from a vector database (Weaviate in this case).

### Reflection Agent

- **`use_reflection_wf.py`**: Illustrates the `ReflectionAgent`, which incorporates self-reflection to improve its performance over time. This example demonstrates how to configure the agent's reflection capabilities and observe its behavior in a question-answering scenario.

## Usage

Each example file can be run independently to observe the behavior of the different agent types.

**Note:** For examples using external services (e.g., ScaleSerp, Weaviate), ensure you have the necessary credentials and configurations set up.

## Key Concepts

- **Agents**: Autonomous entities that can reason, act, and learn to achieve goals.
- **Tools**: Extend the capabilities of agents by providing access to external resources or functionalities.
- **LLMs**: Large language models that power the reasoning and language generation capabilities of agents.
- **Workflows**: Define the sequence of steps and interactions within an agent-based system.
- **Callbacks**: Allow you to monitor and interact with the workflow execution process (e.g., tracing).
