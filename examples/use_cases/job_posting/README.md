# Job Posting Example with Planner

This directory contains an example demonstrating how to use Dynamiq agents and a planner to automate the process of creating a job posting. The workflow leverages multiple agents with specialized roles to research the company, write the job description, review and edit the content, and finally generate a well-structured job posting in markdown format.

## Components

### `main.py`

- Defines the workflow logic for creating a job posting.
- Initializes connections to Anthropic and ScaleSerp.
- Creates instances of `ReActAgent` for research, writing, and reviewing.
- Creates a `LinearOrchestrator` to manage the workflow of multiple agents.
- Executes the workflow with a sample input containing company information and hiring needs.

## Workflow Logic

1. The user provides input data containing company information (link, domain, hiring needs, benefits).
2. The `LinearOrchestrator` receives the input and distributes tasks to the agents:
   - **Researcher Analyst:** Analyzes the company website and provided description to extract insights on culture, values, and specific needs.
   - **Job Description Writer:** Uses insights from the Researcher Analyst to create a detailed and engaging job posting.
   - **Job Description Reviewer and Editor:** Reviews the job description for accuracy, engagement, and alignment with the company's values.
3. Each agent utilizes its assigned tools (e.g., ScaleSerp for web search) to complete its task.
4. The `LinearOrchestrator` collects the outputs from each agent and generates a final, formatted job posting in markdown.

## Usage

1. **Set up environment variables:**
   - `ANTHROPIC_API_KEY`: Your Anthropic API key.
   - `SCALESERP_API_KEY`: Your ScaleSerp API key.
2. **Run the workflow:** `python main.py`

## Key Concepts

- **Planner:** A high-level agent that orchestrates the workflow and assigns tasks to specialized agents.
- **Agent Specialization:** Designing agents with specific roles and expertise to handle different aspects of the task.
- **Tool Usage:** Leveraging tools to enhance the capabilities of agents (e.g., web search, file access).
- **Iterative Refinement:** The workflow allows for iterative refinement of the job posting through the reviewer agent.
- **Automated Output:** The final job posting is automatically generated in markdown format, ready for use.
