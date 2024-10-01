# Trip Planner

This project demonstrates the use of AI agents to create a comprehensive trip planning system. It utilizes the Dynamiq framework to orchestrate multiple AI agents for tasks such as city selection, city guide creation, and travel itinerary generation.

## Directory Structure

```
trip_planner/
    prompts.py
    use_planner.py
    use_orchestrator.py
```

## Files Overview

### prompts.py

This file contains functions to create prompts for the AI agents:

- `create_customer_prompt()`: Generates a detailed prompt for comprehensive trip planning, including city selection, guide creation, and itinerary development.
- `create_simple_customer_prompt()`: Creates a simpler prompt focused on city guide creation.

### use_planner.py

This script demonstrates the use of a Linear Orchestrator to manage AI agents for trip planning:

- Utilizes OpenAI or Anthropic language models.
- Implements various tools like web search, web scraping, and calculations.
- Creates specialized agents for city selection, city guide creation, and travel writing.
- Processes user input to generate a detailed travel guide.

### use_orchestrator.py

Similar to `use_planner.py`, but uses an Adaptive Orchestrator instead of a Linear Orchestrator:

- Offers more flexibility in agent interactions and task management.
- Otherwise similar in functionality to `use_planner.py`.

## Key Features

- Multi-agent system for comprehensive trip planning
- Flexible choice between OpenAI (GPT) and Anthropic (Claude) language models
- Integration with external tools for web search and scraping
- Customizable prompts for different levels of detail in trip planning
- Output of detailed travel guides in Markdown format

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

3. Follow the prompts to input your travel details:
   - Your current location
   - Cities you want to visit
   - Travel dates
   - Your interests

4. The system will generate a detailed travel guide, which will be printed to the console and saved as a Markdown file.

## Dependencies

This project relies on the Dynamiq framework and its components. Ensure you have the following installed:

- dynamiq
- openai (if using GPT models)
- anthropic (if using Claude models)
- Other dependencies as required by the Dynamiq framework

## Note

This project is an example of using AI agents for trip planning. The actual performance and accuracy of the generated travel guides depend on the underlying AI models and the quality of data sources accessed by the tools.
