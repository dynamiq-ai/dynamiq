# Literature Overview Generator

This project demonstrates the use of AI agents to create a comprehensive literature overview on a given topic. It utilizes the Dynamiq framework to orchestrate multiple AI agents for tasks such as research and writing, with a focus on generating content for research papers.

## Directory Structure

```
literature_overview/
    use_planner.py
    use_orchestrator.py
```

## Files Overview

### use_planner.py

This script demonstrates the use of a Linear Orchestrator to manage AI agents for literature overview generation:

- Utilizes OpenAI or Anthropic language models.
- Implements various tools including web search, web scraping, arXiv search, and calculations.
- Creates specialized agents for research and writing.
- Processes user input to generate a detailed literature overview on a given topic.

### use_orchestrator.py

Similar to `use_planner.py`, but uses an Adaptive Orchestrator instead of a Linear Orchestrator:

- Offers more flexibility in agent interactions and task management.
- Otherwise similar in functionality to `use_planner.py`.

## Key Features

- Multi-agent system for comprehensive literature overview generation
- Flexible choice between OpenAI (GPT) and Anthropic (Claude) language models
- Integration with external tools for web search, web scraping, and arXiv search
- Customizable content structure
- Output of detailed literature overviews in Markdown format

## Usage

1. Set up the required environment variables:
   - `SERP_API_KEY`: API key for the ScaleSERP tool
   - `ZENROWS_API_KEY`: API key for the ZenRows scraping tool
   - OpenAI or Anthropic API credentials (as per your chosen provider)

2. Run either `use_planner.py` or `use_orchestrator.py`:
   ```
   python use_planner.py
   ```
   or
   ```
   python use_orchestrator.py
   ```

3. The system will generate a detailed literature overview on the topic "LLM based Multi-Agent Systems and Frameworks".

4. The generated overview will be printed to the console and saved as a Markdown file named `article_gpt.md`.

## Overview Structure

The generated literature overviews follow a specific structure:

1. Introduction
2. Main concepts
3. Applications
4. Conclusion
5. Sources

The overview is formatted in Markdown, including proper syntax for headings, lists, and other formatting elements.

## Agents

1. **Research Analyst**: Finds the most relevant and up-to-date information on the requested topic using web search, web scraping, and arXiv search tools.
2. **Writer and Editor**: Creates high-quality content based on the information provided by the Research Analyst.

## Tools

- ScaleSERPTool: For web searches
- ZenRowsTool: For web scraping

## Dependencies

This project relies on the Dynamiq framework and its components. Ensure you have the following installed:

- dynamiq
- openai (if using GPT models)
- anthropic (if using Claude models)
- Other dependencies as required by the Dynamiq framework

## Note

This project is an example of using AI agents for literature overview generation. The actual performance and accuracy of the generated overviews depend on the underlying AI models and the quality of data sources accessed by the tools. The content is focused on "LLM based Multi-Agent Systems and Frameworks" but can be modified to cover other topics by changing the `user_prompt` in the scripts.
