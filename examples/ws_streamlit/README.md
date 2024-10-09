# Streamlit WebSocket Examples

This directory contains examples demonstrating how to build interactive Streamlit applications that communicate with Dynamiq workflows via WebSockets. These examples showcase streaming capabilities and provide a foundation for creating real-time AI-powered applications.

## Examples

### Basic LLM Streaming

- **`example_llm/`**:
    - **`server.py`**: Defines a FastAPI server that exposes a WebSocket endpoint for interacting with a simple Dynamiq workflow consisting of an OpenAI LLM node. It handles incoming messages, runs the workflow, and streams the LLM's output back to the client.
    - **`app.py`**: Implements a Streamlit application that connects to the WebSocket server, sends user prompts, and displays the streaming LLM response in a chat-like interface.

### Agent with Tools and Streaming

- **`example_agent/`**:
    - **`server.py`**: Similar to `example_llm/server.py`, but this example uses a ReAct agent equipped with a ScaleSerp tool for web search. It demonstrates how to integrate tools into agent workflows and stream the agent's reasoning and actions.
    - **`app.py`**: A Streamlit application that interacts with the agent server, allowing users to chat with the agent and observe its tool usage in real-time.

### Agent with Memory and Streaming

- **`example_agent_chat/`**:
    - **`server.py`**: This example showcases a SimpleAgent with in-memory storage for conversation history. It demonstrates how to maintain context across multiple user interactions and stream the agent's responses.
    - **`app.py`**: A Streamlit application that connects to the agent server, enabling users to engage in a chat with the agent, which remembers previous interactions thanks to its memory.

## Usage

1. **Set up environment variables:** Ensure you have the `OPENAI_API_KEY` environment variable set with your OpenAI API key.
2. **Run the server:** Navigate to the desired example directory (e.g., `example_llm/`) and run `python server.py`. This will start the FastAPI server.
3. **Run the Streamlit app:** In a separate terminal, navigate to the same example directory and run `streamlit run app.py`. This will open the Streamlit application in your browser.

## Key Concepts

- **WebSockets**: Enables bi-directional, real-time communication between the Streamlit application and the Dynamiq workflow server.
- **Streaming**: Allows the LLM or agent's output to be displayed incrementally as it is generated, providing a more interactive user experience.
- **Agents**: Autonomous entities that can interact with their environment (e.g., using tools, accessing memory) to achieve goals.
- **Tools**: Extend the capabilities of agents by providing access to external resources or functionalities.
- **Memory**: Enables agents to store and retrieve information, allowing them to maintain context and learn from past interactions.
