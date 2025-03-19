from dynamiq.connections.connections import Firecrawl
from dynamiq.nodes.tools.firecrawl import FirecrawlTool

if __name__ == "__main__":

    connection = Firecrawl()

    tool = FirecrawlTool(connection=connection)

    input_data = {
        "url": "https://example.com",
    }

    result = tool.run(input_data)

    print(result)
