"""
Manus Assistant Agent

A versatile AI assistant with context compaction capabilities, web search,
file management, todo tracking, and content generation tools.

Features:
- Context Compaction: Automatically manages conversation context when it grows too large
- Todo Management: Track tasks with statuses (pending, in_progress, completed)
  - Todos are stored as a JSON file in the shared file store
- Web Search: Search the web using Tavily for up-to-date information
- Web Scraping: Extract content from web pages using Firecrawl
- File Operations: Read, write, list, and manage files in storage
- Content Generation: Generate and save various types of content

Usage:
    from examples.use_cases.manus_assistant.agent import create_agent, run_workflow

    # Simple usage
    result = run_workflow("Research the latest AI developments and save a summary")

    # With files
    files = [read_file_as_bytesio("data.csv", "data.csv")]
    result = run_workflow("Analyze this data", files_to_upload=files)

    # With todo tracking enabled
    result = run_workflow(
        "Create a research plan, track each step, and save the results",
        todo_enabled=True
    )
"""

import io
from typing import Any

from dynamiq.connections import Firecrawl, Tavily
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.agents.utils import SummarizationConfig
from dynamiq.nodes.tools.firecrawl import FirecrawlTool
from dynamiq.nodes.tools.tavily import TavilyTool
from dynamiq.nodes.types import Behavior, InferenceMode
from dynamiq.storages.file import InMemoryFileStore
from dynamiq.storages.file.base import FileStoreConfig
from dynamiq.utils.logger import logger
from examples.llm_setup import setup_llm

AGENT_ROLE = """You are Manus, an advanced AI assistant with exceptional capabilities in research,
analysis, and content creation. You have access to powerful tools for web search, web scraping,
file management, and task tracking.

## Your Core Capabilities:

1. **Task Management**: Track and manage tasks using the todo system
   - Create todos for multi-step tasks
   - Update status as you progress (pending → in_progress → completed)
   - Monitor completion rates

2. **Web Research**: Search the internet for up-to-date information using Tavily search

3. **Web Scraping**: Extract detailed content from web pages using Firecrawl

4. **File Management**: Read, write, and list files in your working storage

5. **Content Generation**: Create reports, summaries, analyses, and other documents

## Guidelines:

### Task Management:
- For complex tasks, create a todo list to track progress
- Update todo status as you work through tasks
- Use the todo-read tool to check current status
- Use the todo-write tool to save/update your todo list

### Research & Information Gathering:
- When asked to research a topic, use web search to find relevant, current information
- For detailed content extraction from specific URLs, use the Firecrawl scraping tool
- Synthesize information from multiple sources for comprehensive answers
- Always cite your sources when providing research findings

### File Operations:
- Use file-list to see what files are available in storage
- Use file-reader to read and analyze file contents
- Use file-writer to save generated content, reports, or analysis results
- When saving files, use descriptive filenames that reflect the content

### Context Management:
- You have automatic context compaction enabled
- When context grows large, the system will automatically summarize previous interactions
- Important information will be preserved, but be concise in your responses

## Response Format:
- Provide clear, actionable responses
- Break down complex tasks into steps
- Explain your reasoning when making decisions
- Always confirm successful file operations
"""


def read_file_as_bytesio(file_path: str, filename: str | None = None, description: str | None = None) -> io.BytesIO:
    """
    Reads the content of a file and returns it as a BytesIO object.

    Args:
        file_path (str): The path to the file.
        filename (str, optional): Custom filename for the BytesIO object.
        description (str, optional): Custom description for the BytesIO object.

    Returns:
        io.BytesIO: The file content in a BytesIO object with custom attributes.
    """
    with open(file_path, "rb") as f:
        file_content = f.read()

    file_io = io.BytesIO(file_content)
    file_io.name = filename if filename else file_path.split("/")[-1]
    file_io.description = description if description else "Uploaded file"

    return file_io


def create_agent(
    model_provider: str = "gpt",
    model_name: str = "gpt-4o",
    temperature: float = 0.1,
    max_loops: int = 30,
    context_compaction_enabled: bool = True,
    todo_enabled: bool = True,
) -> Agent:
    """
    Create and configure the Manus Assistant agent with necessary tools.

    Args:
        model_provider (str): The LLM provider ("gpt", "claude", etc.)
        model_name (str): The model name to use
        temperature (float): Temperature for LLM generation
        max_loops (int): Maximum reasoning loops for the agent
        context_compaction_enabled (bool): Enable automatic context summarization
        todo_enabled (bool): Enable todo list management tools

    Returns:
        Agent: A configured Manus Assistant Agent ready to run.
    """
    # Setup LLM
    llm = setup_llm(
        model_provider=model_provider,
        model_name=model_name,
        temperature=temperature,
        max_tokens=4096,
    )

    # Create shared file store config (handles files AND todos)
    file_store_config = FileStoreConfig(
        enabled=True,
        backend=InMemoryFileStore(),
        agent_file_write_enabled=True,
        todo_enabled=todo_enabled,  # Todos stored in ._agent/todos.json
    )

    # Initialize external tools (web search, scraping)
    tools = []

    # Web Search Tool - Tavily
    try:
        tool_search = TavilyTool(
            name="web-search",
            connection=Tavily(),
            search_depth="advanced",
            max_results=10,
            include_answer=True,
        )
        tools.append(tool_search)
        logger.info("Tavily search tool initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize Tavily search tool: {e}")

    # Web Scraping Tool - Firecrawl
    try:
        tool_scrape = FirecrawlTool(
            name="web-scraper",
            connection=Firecrawl(),
        )
        tools.append(tool_scrape)
        logger.info("Firecrawl scraping tool initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize Firecrawl scraping tool: {e}")

    # Configure summarization for context compaction
    summarization_config = SummarizationConfig(
        enabled=context_compaction_enabled,
        context_usage_ratio=0.75,
        max_attempts=3,
    )

    # Create the agent
    # File tools (read, write, list, search) are auto-added from file_store config
    # Todo tools are auto-added when file_store.todo_enabled=True
    agent = Agent(
        name="Manus Assistant",
        llm=llm,
        tools=tools,
        role=AGENT_ROLE,
        max_loops=max_loops,
        inference_mode=InferenceMode.XML,
        behaviour_on_max_loops=Behavior.RETURN,
        summarization_config=summarization_config,
        file_store=file_store_config,
    )

    logger.info(f"Manus Assistant created with {len(agent.tools)} tools")
    return agent


def run_workflow(
    prompt: str,
    files_to_upload: list[io.BytesIO] | None = None,
    model_provider: str = "gpt",
    model_name: str = "gpt-4o",
    context_compaction_enabled: bool = True,
    todo_enabled: bool = True,
) -> tuple[str, dict[str, Any]]:
    """
    Main function to run the Manus Assistant workflow.

    Args:
        prompt (str): The task or question for the agent.
        files_to_upload (List[io.BytesIO], optional): Files to make available to the agent.
        model_provider (str): The LLM provider to use.
        model_name (str): The model name to use.
        context_compaction_enabled (bool): Enable automatic context summarization.
        todo_enabled (bool): Enable todo list management tools.

    Returns:
        tuple[str, dict]: The agent's response content and output files.
    """
    try:
        agent = create_agent(
            model_provider=model_provider,
            model_name=model_name,
            context_compaction_enabled=context_compaction_enabled,
            todo_enabled=todo_enabled,
        )

        input_data = {"input": prompt}
        if files_to_upload:
            input_data["files"] = files_to_upload

        result = agent.run(input_data=input_data)

        content = result.output.get("content", "")
        files = result.output.get("files", {})

        logger.info(f"Workflow completed. Response length: {len(content)}")
        return content, files

    except Exception as e:
        logger.error(f"Workflow error: {e}")
        return f"Error during execution: {str(e)}", {}


def main():
    """Example usage of the Manus Assistant."""
    print("=" * 60)
    print("Manus Assistant - Demo")
    print("=" * 60)

    prompt = """
    I need you to research the latest news about AI agents.

    Please:
    1. Create a todo list to track your progress
    2. Search for the latest news
    3. Provide a brief summary of the top 3 developments
    4. Update your todo list as you complete each step
    """

    print(f"\nTask: {prompt.strip()}")
    print("-" * 60)

    content, files = run_workflow(prompt, todo_enabled=True)

    print("\nResponse:")
    print(content)

    if files:
        print("\nGenerated Files:")
        for filename in files.keys():
            print(f"  - {filename}")


if __name__ == "__main__":
    main()
