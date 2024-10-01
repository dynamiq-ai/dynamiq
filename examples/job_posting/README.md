# Job Posting Workflow

This project demonstrates an AI-powered job posting workflow using different orchestration strategies. It showcases the use of multiple AI agents to research, write, and review job descriptions based on company information and hiring needs.

## Project Structure

```
job_posting/
    use_planner.py
    use_orchestrator.py
```

## Components

### 1. Linear Orchestrator (use_planner.py)

This script implements a job posting workflow using a LinearOrchestrator. It includes the following components:

- ManagerAgent: Oversees the workflow and delegates tasks to specialized agents
- ReActAgent: Implements agents for research, writing, and reviewing
- LinearOrchestrator: Manages the workflow execution in a linear fashion
- Various tools: ScaleSERPTool, ZenRowsTool, CalculatorTool, ScrapeSummarizerTool, FileReadTool

### 2. Adaptive Orchestrator (use_orchestrator.py)

This script implements a job posting workflow using an AdaptiveOrchestrator. It includes similar components to the linear orchestrator but uses a more flexible execution strategy.

## Setup and Usage

1. Install the required dependencies (dynamiq and its components).
2. Set up API keys for OpenAI, Anthropic, SERP, and ZenRows as environment variables.
3. Create a file named `job_example.md` in the same directory (used by the FileReadTool).
4. Run either script:
   - For linear orchestration: `python job_posting/use_planner.py`
   - For adaptive orchestration: `python job_posting/use_orchestrator.py`

## Workflow Overview

1. Both scripts initialize three specialized agents:
   - Researcher Analyst: Gathers information about the company and job
   - Job Description Writer: Creates the job posting content
   - Job Description Reviewer and Editor: Reviews and refines the job posting
2. A ManagerAgent oversees the workflow and delegates tasks to the appropriate agent.
3. The workflow is executed using either a LinearOrchestrator or an AdaptiveOrchestrator.
4. The system generates a job posting based on the provided company information and hiring needs.
5. The final job posting is printed to the console and saved to a markdown file (`job_generated.md` or `job_generated_.md`).

## Customization

You can modify the scripts to:
- Change the language model provider (OpenAI or Anthropic)
- Adjust the agent configurations and tools
- Modify the input prompts and company information

## Key Differences Between Orchestrators

1. LinearOrchestrator (use_planner.py):
   - Executes agents in a predefined, sequential order
   - Suitable for workflows with a clear, linear progression

2. AdaptiveOrchestrator (use_orchestrator.py):
   - Allows for more flexible execution of agents
   - Can adapt the workflow based on intermediate results or changing requirements
   - Potentially more efficient for complex or dynamic workflows

## Note

This project is for demonstration purposes and showcases how AI agents can be used to create a job posting workflow. The actual content generation and quality may vary based on the input provided and the capabilities of the underlying language models.
