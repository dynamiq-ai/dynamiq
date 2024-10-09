# Tools Examples

This directory contains examples demonstrating how to use and integrate various tools within Dynamiq workflows. Tools extend the capabilities of agents by providing access to external resources or functionalities, such as web search, code execution, file reading, and API interactions.

## Examples

### Web Search and Scraping

- **`use_react_search.py`**: Demonstrates using a `ReActAgent` with a `ScaleSerpTool` for web search. The agent answers user queries by retrieving and summarizing information from search results.
- **`scraper.py`**: Implements a `ScraperSummarizerTool` that combines web scraping using ZenRows API and content summarization using an LLM.
- **`search_engine.py`**: Defines a `SearchEngineTool` that combines web searching, content scraping, and summarization into a single tool.

### Code Execution

- **`use_react_with_coding.py`**: Showcases a `ReActAgent` equipped with an `E2BInterpreterTool` for executing Python code. The agent solves coding tasks and provides results.

### API Interaction

- **`use_http_api_node.py`**: Demonstrates using the `HttpApiCall` node to interact with an external API (Cat Fact API in this case).

### File Reading

- **`file_reader.py`**: Implements a `FileReadTool` for reading the content of a file from local storage.

### Firecrawl Integration

- **`use_firecrawl.py`**: Demonstrates using the `FirecrawlTool` to interact with the Firecrawl API for website analysis.

### Function Tools

- **`use_function_tool.py`**: Showcases how to create and use function tools, both with and without the `@function_tool` decorator.
- **`use_react_fc.py`**: Demonstrates using a `ReActAgent` with a custom function tool (`calculate_age`) and a `ScaleSerpTool` to answer a query involving age calculation and web search.

### Mathematical Evaluation

- **`calculator_sympy.py`**: Implements a `CalculatorTool` using SymPy to evaluate mathematical expressions.

## Usage

Each example file can be run independently to observe the functionality of the respective tool.

**Note:** For examples using external services (e.g., ScaleSerp, ZenRows, Firecrawl), ensure you have the necessary credentials and configurations set up.

## Key Concepts

- **Tools**: Extend agent capabilities by providing access to external resources or functionalities.
- **Web Search and Scraping**: Tools for retrieving and processing information from the web.
- **Code Execution**: Tools for executing code in various programming languages.
- **API Interaction**: Tools for interacting with external APIs.
- **File Reading**: Tools for accessing and processing data from files.
- **Function Tools**: Tools that encapsulate custom functions for specific tasks.
- **Mathematical Evaluation**: Tools for performing mathematical calculations and symbolic manipulation.
