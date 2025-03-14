from typing import Any, Literal

from pydantic import ConfigDict, Field

from dynamiq.connections import ZenRows
from dynamiq.nodes import ErrorHandling, Node, NodeGroup
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.tools.zenrows import ZenRowsInputSchema, ZenRowsTool
from dynamiq.prompts import Message, Prompt
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

PROMPT_TEMPLATE_SUMMARIZER_LITE = """
Provide the human readable clean content, keep all details that are relevant, remove all redundant information, extra additional marks and symbols. "
Where it is possible, provide some summary, always try to keep exact information with numbers, etc\n\n"
CONTENT\n----------\n{chunk}"
"""  # noqa


class ScraperSummarizerTool(ZenRowsTool):
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "Scraper-Summarizer"
    description: str = (
        "A tool for scraping webpages using ZenRows API and summarizing the content. " "Input should be a valid URL."
    )
    connection: ZenRows
    llm: Node
    error_handling: ErrorHandling = Field(default_factory=lambda: ErrorHandling(timeout_seconds=600))
    chunk_size: int = 8000
    prompt: str = PROMPT_TEMPLATE_SUMMARIZER_LITE

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {"llm": True}

    def to_dict(self, **kwargs) -> dict:
        data = super().to_dict(**kwargs)
        data["llm"] = self.llm.to_dict(**kwargs)
        return data

    def generate_prompt(self, chunk: str) -> str:
        return self.prompt.format(chunk=chunk)

    def execute(self, input_data: ZenRowsInputSchema, config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        logger.debug(f"Tool {self.name} - {self.id}: started with input data {input_data}")

        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        url = input_data.url

        params = {
            "url": url,
            "markdown_response": str(self.markdown_response).lower(),
        }

        try:
            response = self.client.request(
                method=self.connection.method,
                url=self.connection.url,
                params=self.connection.params | params,
            )
            response.raise_for_status()
            scrape_result = response.text
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to get results. Error: {e}")
            return {"error": f"Failed to scrape the webpage. {str(e)}"}

        content_chunks = [scrape_result[i : i + self.chunk_size] for i in range(0, len(scrape_result), self.chunk_size)]
        summaries = []
        for chunk in content_chunks:
            current_prompt = self.generate_prompt(chunk)
            result = self.llm.run(
                input_data={},
                prompt=Prompt(messages=[Message(role="user", content=current_prompt)]),
            )
            summaries.append(result.output["content"])

        search_summary = "Source URL:\n" + url + "\nSummary\n" + "\n".join(summaries)

        logger.debug(f"Tool {self.name} - {self.id}: finished with result {search_summary[:50]}")
        return {"url": url, "content": search_summary}
