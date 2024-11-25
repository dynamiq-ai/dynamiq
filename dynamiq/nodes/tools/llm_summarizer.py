from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes import ErrorHandling, Node, NodeGroup
from dynamiq.nodes.node import NodeDependency, ensure_config
from dynamiq.prompts import Message, Prompt
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.utils.logger import logger

PROMPT_TEMPLATE_SUMMARIZER = """
You are tasked with cleaning up and formatting extracted text from an HTML file. The text contains content from various HTML elements like paragraphs, headers, and tables, but without any HTML tags. Your goal is to produce a well-written, coherent piece of content by removing unnecessary information and formatting the remaining text.

Here is the extracted text from the HTML file:

<html_text>
{input}
</html_text>

Follow these steps to clean up and format the text:

1. Identify and remove unnecessary content:
   - Remove any navigation-related text (e.g., "Home", "Next", "Previous", "Go back")
   - Remove button text (e.g., "Click here", "Submit", "Download")
   - Remove footer information (e.g., copyright notices, contact information)
   - Remove any text related to website functionality (e.g., "Search", "Login", "Register")

2. Clean up the remaining content:
   - Remove excessive whitespace, including multiple consecutive blank lines
   - Fix any obvious spelling or grammatical errors
   - Ensure proper capitalization for sentences and proper nouns

3. Format the content:
   - Identify main headings and subheadings, and ensure they are properly formatted
   - Group related paragraphs together
   - If there are lists or tables, ensure they are properly formatted and aligned

4. Final review:
   - Read through the cleaned-up content to ensure it flows logically and coherently
   - Make any necessary adjustments to improve readability and clarity

5. Output the final cleaned and formatted text:
   - Present the cleaned-up content in a well-organized manner
   - Use appropriate line breaks and spacing to enhance readability

Ensure that only the relevant, informative content remains, and that it is presented in a clear, readable format.
"""  # noqa E501


class SummarizerInputSchema(BaseModel):
    input: str = Field(..., description="Parameter to provide text to summarize and clean.")


class SummarizerTool(Node):
    """
    A tool for summarizing and cleaning up text extracted from HTML.

    This tool processes input text, typically extracted from HTML, by removing unnecessary content,
    cleaning up the remaining text, and formatting it into a coherent and well-organized summary.

    Attributes:
        group (Literal[NodeGroup.TOOLS]): The group this node belongs to.
        name (str): The name of the tool.
        description (str): A description of the tool's functionality.
        llm (Node): The language model node used for text processing.
        chunk_size (int): The maximum number of words in each chunk for processing.
        error_handling (ErrorHandling): Configuration for error handling.
        prompt_template (str): The prompt template used for text summarization.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "Summarizer Tool"
    description: str = (
        "A tool for summarizing and cleaning up text extracted from HTML. "
    )
    llm: Node
    chunk_size: int = Field(default=4000, description="The maximum number of words in each chunk")
    error_handling: ErrorHandling = Field(default_factory=lambda: ErrorHandling(timeout_seconds=600))
    prompt_template: str = Field(
        default=PROMPT_TEMPLATE_SUMMARIZER,
        description="The prompt template for the summarizer",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)
    input_schema: ClassVar[type[SummarizerInputSchema]] = SummarizerInputSchema

    def init_components(self, connection_manager: ConnectionManager | None = None) -> None:
        """
        Initialize the components of the tool.

        Args:
            connection_manager (ConnectionManager, optional): connection manager. Defaults to ConnectionManager.
        """
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        if self.llm.is_postponed_component_init:
            self.llm.init_components(connection_manager)

    def reset_run_state(self):
        """
        Reset the intermediate steps (run_depends) of the node.
        """
        self._run_depends = []

    @property
    def to_dict_exclude_params(self) -> dict:
        """
        Property to define which parameters should be excluded when converting the class instance to a dictionary.

        Returns:
            dict: A dictionary defining the parameters to exclude.
        """
        return super().to_dict_exclude_params | {"llm": True}

    def to_dict(self, **kwargs) -> dict:
        """
        Convert the tool to a dictionary representation.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary representation of the tool.
        """
        data = super().to_dict(**kwargs)
        data["llm"] = self.llm.to_dict(**kwargs)
        return data

    def _process_chunk(self, chunk: str, config: RunnableConfig, **kwargs) -> str:
        """
        Process a single chunk of text using the language model.

        Args:
            chunk (str): The text chunk to process.
            config (RunnableConfig): The configuration for running the model.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The processed text chunk.

        Raises:
            ValueError: If the language model execution fails.
        """
        prompt = self.prompt_template.format(input=chunk)
        result = self.llm.run(
            input_data={},
            prompt=Prompt(messages=[Message(role="user", content=prompt)]),
            config=config,
            **(kwargs | {"parent_run_id": kwargs.get("run_id")}),
        )
        self._run_depends = [NodeDependency(node=self.llm).to_dict()]
        if result.status != RunnableStatus.SUCCESS:
            raise ValueError("LLM execution failed")
        return result.output["content"]

    def execute(
        self, input_data: SummarizerInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Execute the summarization tool on the input data.

        This method processes the input text, potentially breaking it into chunks if it exceeds
        the specified chunk size, and then summarizes each chunk using the language model.

        Args:
            input_data (dict[str, Any]): A dictionary containing the input text under the 'input' key.
            config (RunnableConfig, optional): The configuration for running the tool.
            **kwargs: Additional keyword arguments.

        Returns:
            dict[str, Any]: A dictionary containing the summarized content under the 'content' key.

        Raises:
            ValueError: If the input_data does not contain an 'input' key.
        """
        config = ensure_config(config)
        self.reset_run_state()
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        input_text = input_data.input
        logger.debug(
            f"Tool {self.name} - {self.id}: started with input text length: {len(input_text)}, "
            f"word count: {len(input_text.split())}"
        )

        words = input_text.split()
        if len(words) > self.chunk_size:
            content_chunks = [
                " ".join(words[i : i + self.chunk_size])
                for i in range(0, len(words), self.chunk_size)
            ]
            summaries = [self._process_chunk(chunk, config, **kwargs) for chunk in content_chunks]
            summary = "\n".join(summaries)
        else:
            summary = self._process_chunk(input_text, config, **kwargs)

        logger.debug(
            f"Tool {self.name} - {self.id}: finished with result length: {len(summary)}, "
            f"word count: {len(summary.split())}"
        )
        return {"content": summary}
