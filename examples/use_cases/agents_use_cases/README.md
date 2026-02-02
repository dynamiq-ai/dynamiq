# Agent Use Case Examples

This directory contains examples demonstrating how to apply Dynamiq agents to solve real-world problems in specific domains.

## Examples

### Agent Analyst with Files

- **`house_price_regression.py`**: Showcases an agent that can analyze data from CSV files. This example demonstrates how to:
    - Upload files to the agent's environment using the `E2BInterpreterTool`.
    - Provide file descriptions to help the agent understand the data structure.
    - Craft prompts that instruct the agent to perform specific data analysis tasks (e.g., linear regression).
    - Retrieve and interpret the agent's results.

** Ensure you have the necessary API keys: **
   - `OPENAI_API_KEY`: Your OpenAI API key.
   - `E2B_API_KEY`: Your E2B API key.

### Humanizer skill (Dynamiq API)

- **`agent_humanizer_skill.py`**: Uses the **humanizer** skill from the Dynamiq API (skill id `cfb2ddd9-7b5b-457f-9ef6-666db9c45eb3`, version `adcf8695-31b0-4a50-ba68-3606b5158e7c`). The agent uses SkillsTool to list and get the humanizer skill content, then applies its guidelines to remove signs of AI-generated writing and make text sound more natural. Set `DYNAMIQ_URL` (e.g. `https://api.sandbox.getdynamiq.ai`) and `DYNAMIQ_API_KEY`.

### Financial Agent

- **`agent_financial.py`**: Demonstrates an agent that can access and process financial information. This example illustrates how to:
    - Configure an agent with tools for accessing external APIs (e.g., for retrieving cryptocurrency prices).
    - Define a goal that encourages the agent to generate and execute code to solve the user's query.
    - Observe the agent's reasoning process and the final answer it provides.

** Ensure you have the necessary API keys: **
   - `OPENAI_API_KEY`: Your OpenAI API key.
   - `E2B_API_KEY`: Your E2B API key.

## Usage

Each example file can be run independently to observe the agent's behavior in the respective use case.

**Note:** For examples using external services (e.g., E2B), ensure you have the necessary credentials and configurations set up. You may also need to replace placeholder file paths and descriptions with your own data.

## Key Concepts

- **Domain-Specific Agents**: Agents tailored to solve problems in specific areas (e.g., data analysis, finance).
- **Tool Integration**: Leveraging tools to provide agents with access to relevant data and functionalities.
- **Prompt Engineering**: Crafting effective prompts to guide the agent's reasoning and actions.
- **Result Interpretation**: Understanding and interpreting the agent's output in the context of the use case.
