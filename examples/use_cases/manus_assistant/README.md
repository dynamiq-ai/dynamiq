# Manus Assistant

A versatile AI assistant with context compaction capabilities, web search, file management, and content generation tools.

## Features

- **Context Compaction**: Automatically manages conversation context when it grows too large, preventing token limit issues while preserving important information
- **Web Search**: Search the web using Tavily for up-to-date information
- **Web Scraping**: Extract content from web pages using Firecrawl
- **File Operations**: Read, write, and list files in an in-memory storage system
- **Content Generation**: Generate and save various types of content including reports, summaries, and analyses

## Prerequisites

Set the following environment variables:

```bash
# Required for LLM
export OPENAI_API_KEY="your-openai-api-key"
# OR for Claude
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Required for web search
export TAVILY_API_KEY="your-tavily-api-key"

# Optional for web scraping
export FIRECRAWL_API_KEY="your-firecrawl-api-key"
```

## Usage

### Basic Usage

```python
from examples.use_cases.manus_assistant.agent import run_workflow

# Simple research task
result, files = run_workflow(
    "Research the latest developments in AI agents and summarize the findings"
)
print(result)
```

### With File Upload

```python
from examples.use_cases.manus_assistant.agent import run_workflow, read_file_as_bytesio

# Upload a file for analysis
files = [read_file_as_bytesio("data.csv", "sales_data.csv", "Q4 sales data")]

result, output_files = run_workflow(
    "Analyze this sales data and create a summary report",
    files_to_upload=files
)
print(result)
```

### Custom Configuration

```python
from examples.use_cases.manus_assistant.agent import create_agent, create_file_store

# Create with custom settings
file_store = create_file_store()

agent = create_agent(
    model_provider="claude",
    model_name="claude-3-5-sonnet-20240620",
    temperature=0.2,
    max_loops=50,
    context_compaction_enabled=True,
    file_store=file_store,
)

# Run the agent directly
result = agent.run(input_data={"input": "Your task here"})
```

## Tools Available

### Web Search (Tavily)
```python
# The agent can search for information:
"Search for recent news about quantum computing"
```

### Web Scraping (Firecrawl)
```python
# Extract content from specific URLs:
"Scrape the content from https://example.com/article and summarize it"
```

### File Operations
```python
# List files:
"List all files in the storage"

# Read files:
"Read the contents of report.pdf"

# Write files:
"Save this analysis as analysis_report.md"
```

## Context Compaction

The agent includes automatic context compaction powered by `SummarizationConfig`:

- **Enabled by default**: Automatically summarizes conversation history when context grows large
- **Configurable threshold**: Triggers at 75% of the context window by default
- **Preserves important info**: Key facts and findings are maintained in summaries

This allows the agent to handle long conversations and complex multi-step tasks without running into token limits.

## Example Tasks

1. **Research and Report**:
   ```
   Research the top 5 AI startups of 2024, compile their funding information,
   and save a detailed report as ai_startups_2024.md
   ```

2. **Data Analysis**:
   ```
   Read the uploaded CSV file, analyze the trends,
   and create a summary with key insights
   ```

3. **Content Extraction**:
   ```
   Scrape the documentation from https://docs.example.com,
   extract the getting started guide, and save it locally
   ```

4. **Multi-step Research**:
   ```
   Search for information about sustainable energy solutions,
   find the top 3 companies, scrape their websites for details,
   and compile everything into a comprehensive report
   ```

## Running the Demo

```bash
cd /path/to/dynamiq
python -m examples.use_cases.manus_assistant.agent
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Manus Assistant                         │
├─────────────────────────────────────────────────────────────┤
│  LLM (GPT-4o / Claude)                                      │
│  └── Inference Mode: XML                                    │
│  └── Context Compaction: Enabled                            │
├─────────────────────────────────────────────────────────────┤
│  Tools:                                                      │
│  ├── web-search (TavilyTool)      - Web search              │
│  ├── web-scraper (FirecrawlTool)  - Web content extraction  │
│  ├── file-reader (FileReadTool)   - Read files              │
│  ├── file-writer (FileWriteTool)  - Write/create files      │
│  └── file-list (FileListTool)     - List available files    │
├─────────────────────────────────────────────────────────────┤
│  Storage:                                                    │
│  └── InMemoryFileStore            - Temporary file storage  │
└─────────────────────────────────────────────────────────────┘
```
