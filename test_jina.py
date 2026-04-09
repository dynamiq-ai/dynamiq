"""Quick script to test Jina Reader API using JinaScrapeTool.execute()."""
from dotenv import load_dotenv

from dynamiq.connections import Jina
from dynamiq.nodes.tools.jina import JinaScrapeTool, JinaScrapeInputSchema

load_dotenv()


def test_scrape(url, engine="direct", target_selector=None, label=""):
    print(f"\n{'='*60}")
    print(f"Test: {label}")
    print(f"URL: {url}")
    print(f"Engine: {engine}, Selector: {target_selector}")
    print(f"{'='*60}")

    tool = JinaScrapeTool(
        connection=Jina(),
        engine=engine,
    )

    input_data = JinaScrapeInputSchema(
        url=url,
        target_selector=target_selector,
    )

    try:
        result = tool.execute(input_data=input_data)
        content = result.get("content", "")
        print(f"Content length: {len(str(content))}")
        print(f"Preview: {str(content)[:500]}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")


# Test 1: YouTube with direct engine + selector (the failing case)
test_scrape(
    "https://www.youtube.com/@Cristiano/about",
    engine="direct",
    target_selector="#subscriber-count",
    label="YouTube direct + selector (original failing request)",
)
