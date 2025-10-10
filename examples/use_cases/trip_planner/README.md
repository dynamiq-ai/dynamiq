# Trip Planner Example

This directory contains a single example that demonstrates how to coordinate Dynamiq agents as tools to research and assemble a multi-day trip guide.

## Components

- **`prompts.py`** – helper utilities that validate input data and build the customer prompt handed to the manager agent.
- **`agent_trip_planner.py`** – the main entry point that wires a manager agent with three specialized sub-agents (city selection, city guide, itinerary writer) and runs the workflow.

## How It Works

1. The script provisions an LLM through `examples.llm_setup.setup_llm` (defaults to OpenAI) and a `ScaleSerpTool` for web search.
2. Three sub-agents are instantiated with clear roles and instructions to expect `{'input': ...}` payloads when invoked as tools.
3. A manager agent receives the traveler request, delegates work to the sub-agents (optionally in parallel), and synthesizes their findings into a final itinerary.
4. The generated itinerary is printed to the console and saved to `trip_plan.md`.

## Prerequisites

Set the required API keys before running the example:

- `OPENAI_API_KEY` (or the key for the provider configured in `examples/llm_setup.py`)
- `SCALESERP_API_KEY`
- Optionally `ANTHROPIC_API_KEY` if you switch the model provider to Anthropic

## Usage

```bash
cd examples/use_cases/trip_planner
python agent_trip_planner.py
```

You will be prompted for the traveler location, candidate cities, trip dates, and interests. Adjust the default LLM provider or model by editing `examples/llm_setup.py` or by supplying arguments to `run_trip_planner` if you import the helper in your own scripts.
