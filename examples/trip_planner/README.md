# Trip Planner Examples

This directory contains examples demonstrating how to use Dynamiq agents and orchestrators to build a trip planning application. The examples showcase different approaches to orchestrating agents and utilizing tools for research and content generation.

## Components

### `prompts.py`

- Defines functions for generating customer prompts for trip planning.
- Includes functions for validating input data and formatting prompts.

## Examples

### Using Planner (Linear Orchestrator)

- **`use_planner.py`**: Demonstrates a linear workflow using a `LinearOrchestrator` to manage a sequence of agents:
    - **City Selection Expert:** Analyzes travel data to select the best city based on criteria like weather, events, and costs.
    - **City Guide Expert:** Gathers information about the chosen city, including attractions, customs, and recommendations.
    - **City Guide Writer:** Creates a detailed travel guide based on the gathered information.
    - The workflow takes user input for trip details and saves the final output to a markdown file.

### Using Adaptive Orchestrator

- **`use_orchestrator.py`**: Showcases an adaptive workflow using an `AdaptiveOrchestrator` to dynamically manage agent execution:
    - **City Selection Expert:** Similar to the previous example, analyzes travel data to select the best city.
    - **City Guide Expert:** Gathers information about the chosen city, including attractions, customs, and recommendations.
    - **City Guide Writer:** Creates a detailed travel guide based on the gathered information.
    - The `AdaptiveOrchestrator` decides which agent to execute next based on the current state of the workflow.
    - The workflow takes user input for trip details and saves the final output to a markdown file.

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
- **Research Tools:** Utilizing tools like `ScaleSerpTool` and `ZenRowsTool` to gather information from the web.
- **Content Generation:** Leveraging LLMs to generate well-structured and informative travel guides.
- **Prompt Engineering:** Crafting effective prompts to guide the agents' actions and ensure relevant output.
