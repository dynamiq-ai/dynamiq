# Adaptive Orchestrator Examples for Coding Tasks

This directory contains examples demonstrating how to use the `AdaptiveOrchestrator` for managing agents that perform coding tasks. The `AdaptiveOrchestrator` dynamically determines which agent to execute based on the current state of the workflow and the capabilities of the available agents.

## Examples

### Direct Agent Execution

- **`adaptive_coding.py`**: This example showcases a simplified scenario where the `AdaptiveOrchestrator` directly manages a single `ReActAgent` equipped with a coding tool (`E2BInterpreterTool`). It demonstrates how to set up the orchestrator, agent, and tool, and how to execute the workflow with a coding task as input.

### Workflow Integration

- **`adaptive_coding_workflow.py`**: This example demonstrates how to integrate the `AdaptiveOrchestrator` into a Dynamiq workflow. It shows how to create a workflow with the orchestrator as a node, configure the agent and tools, and execute the workflow with a coding task. It also includes tracing to monitor the workflow execution.

## Usage

1. **Set up environment variables:**
   - `OPENAI_API_KEY`: Your OpenAI API key.
2. **Run the desired example:**
   - For direct agent execution: `python adaptive_coding.py`
   - For workflow integration: `python adaptive_coding_workflow.py`

## Key Concepts

- **Adaptive Orchestration:** Dynamically managing the execution of agents based on the workflow's state and agent capabilities.
- **Coding Agents:** Agents equipped with tools like `E2BInterpreterTool` to execute code and solve programming tasks.
- **Workflow Integration:** Integrating the `AdaptiveOrchestrator` into a larger Dynamiq workflow to manage complex processes.
- **Tracing:** Monitoring the workflow execution to understand agent interactions and identify potential issues.
