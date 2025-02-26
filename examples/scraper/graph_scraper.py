import json
from datetime import datetime

from dynamiq.connections import Firecrawl, Jina, ZenRows
from dynamiq.nodes.agents.orchestrators.graph import GraphOrchestrator, GraphState
from dynamiq.nodes.agents.orchestrators.graph_manager import GraphAgentManager
from dynamiq.nodes.agents.simple import SimpleAgent
from dynamiq.nodes.tools.firecrawl import FirecrawlTool
from dynamiq.nodes.tools.function_tool import function_tool
from dynamiq.nodes.tools.jina import JinaScrapeTool
from dynamiq.nodes.tools.zenrows import ZenRowsTool
from examples.llm_setup import setup_llm

llm = setup_llm(model_provider="gpt", model_name="gpt-4o-mini", temperature=0)

jina_scraper = JinaScrapeTool(
    name="Web Page Scraper",
    description="Scrape content from generic web pages",
    connection=Jina(),
    is_optimized_for_agents=True,
)


firecrawl_tool = FirecrawlTool(
    name="Firecrawl Scraper Tool",
    description="Backup tool for web scraping using Firecrawl.",
    connection=Firecrawl(),
    is_optimized_for_agents=True,
)


zenrows_tool = ZenRowsTool(
    name="Zenrows Scraper Tool",
    description="Backup tool for web scraping using ZenRows.",
    connection=ZenRows(),
    is_optimized_for_agents=True,
)


def unwrap_context(context):
    """If context is nested inside a 'context' key, return that; otherwise, return as is."""
    return context.get("context", context)


@function_tool
def initialize_scraper_state(context, **kwargs):
    """Initialize the scraper state for a generic website."""
    context = unwrap_context(context)
    if "scraper_state" not in context:
        context["scraper_state"] = {
            "visited_urls": [],
            "url_queue": [],
            "extracted_data": [],
            "logs": [],  # log details per page
            "stats": {"start_time": datetime.now().isoformat(), "pages_scraped": 0, "items_found": 0},
        }
    start_url = context.get("start_url")
    if start_url not in context["scraper_state"]["visited_urls"]:
        context["scraper_state"]["url_queue"].append(start_url)
    context["max_pages"] = context.get("max_pages", 5)
    return {
        "result": f"Scraper initialized with start URL: {start_url}. Max pages: {context.get('max_pages', 5)}.",
        "context": context,
    }


@function_tool
def check_url_queue(context, **kwargs):
    """
    Check if there are URLs in the queue to process.
    If the URL queue is empty or maximum pages are reached, signal termination by setting next_state to "Finalize".
    """
    context = unwrap_context(context)
    scraper_state = context["scraper_state"]
    if scraper_state["stats"]["pages_scraped"] >= context.get("max_pages", 5):
        return {
            "result": f"Reached maximum pages ({context.get('max_pages', 5)}).",
            "next_state": "Finalize",
            "context": context,
        }
    if not scraper_state["url_queue"]:
        return {"result": "URL queue is empty.", "next_state": "Finalize", "context": context}
    next_url = scraper_state["url_queue"][0]
    context["current_url"] = next_url
    return {"result": f"Processing URL: {next_url}", "next_state": "Scrape Page", "context": context}


@function_tool
def scrape_page_fallback(context, **kwargs):
    """
    Attempts to scrape the current URL using a fallback strategy.
    Tries JinaScrapeTool first, then FirecrawlTool, and finally ZenRowsTool.
    Expects 'current_url' in context.
    Stores the scraped HTML as both 'scraped_content' and 'processor_input' for downstream processing.
    """
    context = unwrap_context(context)
    url = context.get("current_url")
    if not url:
        if "scraper_state" in context and context["scraper_state"].get("url_queue"):
            url = context["scraper_state"]["url_queue"][0]
        else:
            return {"result": "Error: No URL provided in context.", "context": context}
    try:
        result = zenrows_tool.run(input_data={"url": url}, **kwargs)
        scraped_content = result.output.get("content")
        context["scraped_content"] = scraped_content
        context["processor_input"] = scraped_content
        return {"result": "Scraped with ZenRows.", "context": context}
    except Exception as e:
        print(f"ZenRows failed for {url}: {e}")
        try:
            result = firecrawl_tool.run(input_data={"url": url}, **kwargs)
            scraped_content = result.output.get("content")
            context["scraped_content"] = scraped_content
            context["processor_input"] = scraped_content
            return {"result": "Scraped with Firecrawl.", "context": context}
        except Exception as e2:
            print(f"FirecrawlTool failed for {url}: {e2}")
            try:
                result = jina_scraper.run(input_data={"url": url}, **kwargs)
                scraped_content = result.output.get("content")
                context["scraped_content"] = scraped_content
                context["processor_input"] = scraped_content
                return {"result": "Scraped with Jina.", "context": context}
            except Exception as e3:
                print(f"Jina failed for {url}: {e3}")
                return {"result": f"All scrapers failed for URL: {url}", "context": context}


@function_tool
def export_data(context, export_format="json", **kwargs):
    """
    Exports the accumulated extracted data from the scraper state to JSON or CSV format.
    This acts as an intermediate checkpoint after processing each page.
    """
    context = unwrap_context(context)
    scraper_state = context.get("scraper_state", {})
    extracted_data = scraper_state.get("extracted_data", [])
    if not extracted_data:
        return {"result": "No extracted data available for export.", "context": context}
    if export_format.lower() == "csv":
        import csv
        import io

        output = io.StringIO()
        header = extracted_data[0].keys() if extracted_data else []
        writer = csv.DictWriter(output, fieldnames=header)
        writer.writeheader()
        for row in extracted_data:
            writer.writerow(row)
        exported = output.getvalue()
    else:
        exported = json.dumps(extracted_data, indent=2)
    context["exported_data"] = exported
    return {"result": f"Data exported in {export_format.upper()} format.", "context": context}


# Define agents.
page_processor_agent = SimpleAgent(
    name="Page Processor Agent",
    description="Process scraped page content using LLM prompting.",
    llm=llm,
    role="""You are an expert in processing HTML content.
Your input is provided in the context variable 'processor_input' (which contains the scraped HTML).
Clean and simplify the HTML and extract the main content and links.
Return the processed content.""",
)

data_extractor_agent = SimpleAgent(
    name="Data Extractor Agent",
    description="Extract structured data items from processed page content using LLM prompting.",
    llm=llm,
    role="""You are an expert data extraction specialist.
Extract structured data items (e.g., Title, Description, URL) from the processed HTML.
Return the data as a JSON array.""",
)

pagination_agent = SimpleAgent(
    name="Pagination Agent",
    description="Determine the next page URL for pagination using LLM prompting.",
    llm=llm,
    role="""Analyze the processed content to locate a link for the next page.
If found, return the next page URL; if not, clear the URL queue to indicate completion.""",
)

save_data_agent = SimpleAgent(
    name="Data Storage Agent",
    description="Save extracted data into the scraper state using LLM prompting.",
    llm=llm,
    role="""Append the structured data extracted from the page to the scraper state's 'extracted_data' list.
Return a summary of items saved for this page.""",
)


@function_tool
def finalize(context, **kwargs):
    """Finalizes the process by returning the final context."""
    context = unwrap_context(context)
    return {"result": "Final context returned.", "context": context}


graph_manager = GraphAgentManager(name="Generic Scraping Manager", llm=llm)

generic_scraper = GraphOrchestrator(
    name="Generic Website Scraper",
    manager=graph_manager,
    states=[
        GraphState(
            id="Initialize Scraper",
            name="Initialize Scraper",
            description="Set up the initial state for generic website scraping.",
            tasks=[initialize_scraper_state()],
            next_states=["Check URL Queue"],
        ),
        GraphState(
            id="Check URL Queue",
            name="Check URL Queue",
            description="Verify if there are URLs left in the queue to process.",
            tasks=[check_url_queue()],
            next_states=["Scrape Page", "Finalize"],
            condition=None,
        ),
        GraphState(
            id="Scrape Page",
            name="Scrape Page",
            description="Scrape content from the generic web page using fallback scrapers.",
            tasks=[scrape_page_fallback()],
            next_states=["Process Page"],
        ),
        GraphState(
            id="Process Page",
            name="Process Page",
            description="Process the raw HTML content (from 'scraped_content') to simplify and clean it.",
            tasks=[page_processor_agent],
            next_states=["Extract Data"],
            manager=graph_manager,
        ),
        GraphState(
            id="Extract Data",
            name="Extract Data",
            description="Extract structured data items from the processed page content.",
            tasks=[data_extractor_agent],
            next_states=["Save Data"],
            manager=graph_manager,
        ),
        GraphState(
            id="Save Data",
            name="Save Data",
            description="Store the extracted data into the scraper state and log page details.",
            tasks=[save_data_agent],
            next_states=["Export Data"],
            manager=graph_manager,
        ),
        GraphState(
            id="Export Data",
            name="Export Data",
            description="Export the accumulated extracted data to JSON (or CSV) format as a checkpoint.",
            tasks=[export_data()],
            next_states=["Handle Pagination"],
            manager=graph_manager,
        ),
        GraphState(
            id="Handle Pagination",
            name="Handle Pagination",
            description="Determine the next page URL from the processed content and update the URL queue.",
            tasks=[pagination_agent],
            next_states=["Check URL Queue"],
            manager=graph_manager,
        ),
        GraphState(
            id="Finalize",
            name="Finalize",
            description="Final state: return the final context.",
            tasks=[finalize()],
            next_states=[],
            manager=graph_manager,
        ),
    ],
    initial_state="Initialize Scraper",
)


def scrape_generic_website(
    start_url="https://clutch.co/developers/artificial-intelligence/generative?page=1",
    max_pages=5,
):
    """Scrape data from a generic website using the graph orchestrator with fallback scrapers."""
    context = {"start_url": start_url, "max_pages": max_pages}
    generic_scraper.context = context
    result = generic_scraper.run(input_data={})
    final_context = result.get("context", {})
    if "scraper_state" in final_context:
        extracted_data = final_context["scraper_state"].get("extracted_data", [])
        stats = final_context["scraper_state"].get("stats", {})
        exported = final_context.get("exported_data", "")
        return {"data": extracted_data, "stats": stats, "exported": exported, "total_items": len(extracted_data)}
    return result


if __name__ == "__main__":
    results = scrape_generic_website(max_pages=3)
    print(f"Scraped {results.get('total_items', 0)} items from the generic website.")
    if results.get("exported"):
        print("Exported Data:")
        print(results.get("exported"))
