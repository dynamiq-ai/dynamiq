from dynamiq.connections.connections import Firecrawl
from dynamiq.nodes.tools.firecrawl import Action, FirecrawlTool, LocationSettings


def scrape_energy_outlook(connection: Firecrawl) -> None:
    """Scrape the EIA STEO global oil outlook with structured output."""
    tool = FirecrawlTool(
        connection=connection,
        formats=[
            "markdown",
            "links",
            {
                "type": "json",
                "schema": {
                    "type": "object",
                    "properties": {
                        "headline_findings": {"type": "array", "items": {"type": "string"}},
                        "important_dates": {"type": "array", "items": {"type": "string"}},
                    },
                },
                "prompt": "List key findings and any calendar-sensitive events mentioned in the report.",
            },
        ],
        include_tags=["article", "section"],
        wait_for=1500,
        timeout=60000,
        parsers=["pdf"],
        proxy="auto",
        remove_base64_images=True,
        store_in_cache=False,
        location=LocationSettings(country="US", languages=["en-US"]),
        actions=[
            Action(type="wait", milliseconds=1200),
            Action(type="scroll", direction="down"),
            Action(type="scrape"),
        ],
    )

    result = tool.run(
        {
            "url": "https://www.eia.gov/outlooks/steo/report/global_oil.php",
            "maxAge": 3_600_000,
        }
    )

    print("EIA Global Oil scrape output keys:", result.output["content"].keys())


def scrape_blog_for_agent(connection: Firecrawl) -> None:
    """Scrape a blog post with agent-optimized formatting for reasoning workflows."""
    tool = FirecrawlTool(
        connection=connection,
        is_optimized_for_agents=True,
        formats=[
            "markdown",
            "summary",
            {"type": "screenshot", "fullPage": True, "quality": 80},
        ],
        only_main_content=True,
        mobile=True,
        proxy="stealth",
        actions=[
            Action(type="wait", milliseconds=800),
            Action(type="scroll", direction="down"),
        ],
    )

    result = tool.run(
        {
            "url": "https://www.meyerperin.com/posts/2023-05-23-my-experience-using-large-language-models-to-write-blog-posts.html",  # noqa e501
            "onlyMainContent": True,
        }
    )

    print("Agent-focused response:\n")
    print(result.output["content"])


if __name__ == "__main__":
    firecrawl_connection = Firecrawl()
    scrape_energy_outlook(firecrawl_connection)
    scrape_blog_for_agent(firecrawl_connection)
