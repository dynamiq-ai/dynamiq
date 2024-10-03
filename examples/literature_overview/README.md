# Literature Overview Examples

This directory contains examples demonstrating how to use Dynamiq agents and orchestrators to generate a literature overview on a given topic. The examples showcase different approaches to orchestrating agents and utilizing tools for research and content generation.

## Examples

### Using Planner (Linear Orchestrator)

- **`use_planner.py`**: Demonstrates a linear workflow using a `LinearOrchestrator` to manage a sequence of agents:
    - **Research Analyst:** Uses `ScaleSerpTool` and `ScraperSummarizerTool` to research the topic and gather relevant information.
    - **Writer and Editor:** Creates a literature overview based on the research findings.
    - The workflow saves the final output to a markdown file.

### Using Adaptive Orchestrator

- **`use_orchestrator.py`**: Showcases an adaptive workflow using an `AdaptiveOrchestrator` to dynamically manage agent execution:
    - **Research Analyst:** Similar to the previous example, uses `ScaleSerpTool` for research.
    - **Writer and Editor:** Creates the literature overview.
    - The `AdaptiveOrchestrator` decides which agent to execute next based on the current state of the workflow.
    - The workflow saves the final output to a markdown file.

## Usage

1. **Set up environment variables:**
   - `OPENAI_API_KEY`: Your OpenAI API key.
   - `ANTHROPIC_API_KEY`: Your Anthropic API key.
   - `SCALESERP_API_KEY`: Your ScaleSerp API key.
   - `ZENROWS_API_KEY`: Your ZenRows API key.
2. **Run the desired example:**
   - For the linear workflow: `python use_planner.py`
   - For the adaptive workflow: `python use_orchestrator.py`

## Key Concepts

- **Agent Orchestration:** Managing the execution and interaction of multiple agents to achieve a complex goal.
- **Linear Orchestration:** Agents are executed in a predefined sequence.
- **Adaptive Orchestration:** The order of agent execution is dynamically determined based on the workflow's state.
- **Research Tools:** Utilizing tools like `ScaleSerpTool` and `ScraperSummarizerTool` to gather information from the web.
- **Content Generation:** Leveraging LLMs to generate well-structured and informative content based on research findings.
