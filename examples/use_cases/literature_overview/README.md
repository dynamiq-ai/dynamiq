# Literature Overview Example

This directory provides a single example that shows how Dynamiq agents can be composed as tools to research and draft a literature overview on demand.

## Components

- **`agent_literature_overview.py`** – entry point that wires a manager agent with three sub-agents (research, citation curation, writing) and executes the workflow.
- **`examples/components/tools/custom_tools/scraper.py`** – supplies the `ScraperSummarizerTool` used by the research agent to turn scraped pages into concise notes.

## Workflow

1. An LLM is provisioned through `examples.llm_setup.setup_llm` (defaults to OpenAI) and both `ScaleSerpTool` and `ScraperSummarizerTool` are prepared for evidence gathering.
2. Sub-agents are instantiated with explicit instructions to accept payloads shaped as `{'input': ...}` when invoked as tools.
3. The manager agent receives the user brief, delegates tasks (in parallel when possible), and synthesizes the final markdown report.
4. The completed overview is printed to the console and saved to `literature_overview.md`.

## Prerequisites

Ensure the relevant API keys are exported before running the script:

- `OPENAI_API_KEY` (or the provider configured in `examples/llm_setup.py`)
- `SCALESERP_API_KEY`
- `ZENROWS_API_KEY`
- Optionally `ANTHROPIC_API_KEY` if you switch providers

## Usage

```bash
cd examples/use_cases/literature_overview
python agent_literature_overview.py
```

You will be prompted for the topic and optional focus areas. To reuse the workflow programmatically, import and call `run_literature_overview` with your own prompt data.
