# Job Posting Example

This example shows how to compose Dynamiq agents as tools to research, draft, and polish a job posting end-to-end.

## Components

- **`agent_job_posting.py`** – entry point that wires a manager agent with research, value-proposition, writing, and editing sub-agents, plus optional company brief ingestion.
- **`job_example.md`** – sample company brief that can be fed to the workflow (optional).

## Workflow

1. `examples.llm_setup.setup_llm` provisions the large language model (defaults to OpenAI) and a `ScaleSerpTool` for web research. If a company brief file exists, a `FileReaderTool` is configured as well.
2. Sub-agents are instantiated with instructions to accept payloads shaped as `{'input': ...}` when invoked as tools.
3. The manager agent receives the hiring brief, delegates work (parallelizing research and messaging where helpful), and synthesizes the final markdown job post.
4. The finished posting is printed to the console and saved as `job_posting.md`.

## Prerequisites

Export the required API keys prior to running the script:

- `OPENAI_API_KEY` (or the provider configured in `examples/llm_setup.py`)
- `SCALESERP_API_KEY`
- Optionally `ANTHROPIC_API_KEY` if you switch the provider

## Usage

```bash
cd examples/use_cases/job_posting
python agent_job_posting.py
```

You will be prompted for company details, hiring needs, tone preferences, and an optional path to a company brief (defaults to `job_example.md` when present). To reuse the logic programmatically, import and call `run_job_posting` with your own payload.
