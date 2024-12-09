import concurrent.futures
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.node import Node, NodeDependency, NodeGroup, ensure_config
from dynamiq.prompts import prompts
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types import Document
from dynamiq.utils.logger import logger

DEFAULT_PROMPT = """
Given a query and a passage, determine whether the passage contains an answer to the query
by providing a prediction of either 'Yes' or 'No'.
The query is "{{query}}"
The passage is:
{{passage}}
Write the answer final answer: 'Yes' or 'No'
"""


class LLMDocumentRankerInputSchema(BaseModel):
    query: str = Field(..., description="Parameter to provide query for ranking.")
    documents: list[Document] = Field(..., description="Parameter to provide list of documents.")


class LLMDocumentRanker(Node):
    """
    A Node class for ranking documents using a Large Language Model (LLM).

    This class can use any LLM to rank and select relevant documents based on a query. By default, it utilizes an OpenAI
    language model.

    Attributes:
        group (Literal[NodeGroup.RANKERS]): The group the node belongs to. Default is NodeGroup.RANKERS.
        name (str): The name of the node. Default is "LLMDocumentRanker".
        prompt_template (str): The template for the prompt to be used with the LLM. Default is DEFAULT_PROMPT.
        top_k (int): The number of top documents to return. Default is 5.
        llm (BaseLLM): The LLM instance used for ranking. Default is None.

    Example:

        from dynamiq.nodes.rankers import LLMDocumentRanker
        from dynamiq.types import Document

        # Initialize the ranker
        ranker = LLMDocumentRanker()

        # Example input data
        input_data = {
            "query": "example query",
            "documents": [
                Document(content="Document content", score=0.8, metadata={"date": "01 January, 2022"}),
                Document(content="Document content", score=0.9, metadata={"date": "01 January, 2021"})
            ]
        }

        # Execute the ranker
        output = ranker.execute(input_data)

        # Output will be a dictionary with ranked documents
        print(output)
    """

    group: Literal[NodeGroup.RANKERS] = NodeGroup.RANKERS
    name: str = "LLMDocumentRanker"
    prompt_template: str = DEFAULT_PROMPT
    top_k: int = 5
    llm: Node
    input_schema: ClassVar[type[LLMDocumentRankerInputSchema]] = LLMDocumentRankerInputSchema

    def __init__(self, **kwargs):
        """
        Initializes the LLMDocumentRanker with the given parameters and creates a default LLM node.

        Args:
            **kwargs: Additional keyword arguments to be passed to the parent class constructor.
        """
        super().__init__(**kwargs)
        self._run_depends = []

    def reset_run_state(self):
        """
        Reset the intermediate steps (run_depends) of the node.
        """
        self._run_depends = []

    @property
    def to_dict_exclude_params(self):
        """
        Property to define which parameters should be excluded when converting the class instance to a dictionary.

        Returns:
            dict: A dictionary defining the parameters to exclude.
        """
        return super().to_dict_exclude_params | {"llm": True}

    def to_dict(self, **kwargs) -> dict:
        """Converts the instance to a dictionary.

        Returns:
            dict: A dictionary representation of the instance.
        """
        data = super().to_dict(**kwargs)
        data["llm"] = self.llm.to_dict(**kwargs)
        return data

    def init_components(self, connection_manager: ConnectionManager | None = None):
        """
        Initialize the document ranker component.

        Args:
            connection_manager (ConnectionManager): The connection manager to use. Default is a new instance.
        """
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        if self.llm.is_postponed_component_init:
            self.llm.init_components(connection_manager)

    def execute(
        self, input_data: LLMDocumentRankerInputSchema, config: RunnableConfig = None, **kwargs
    ) -> dict[str, Any]:
        """
        Executes the document ranking process.

        Args:
            input_data (LLMDocumentRankerInputSchema): A dictionary containing the query and documents to be ranked.
            config (RunnableConfig, optional): Configuration for the execution. Default is None.
            **kwargs: Additional keyword arguments.

        Returns:
            dict[str, Any]: A dictionary containing the original query and the ranked documents.

        Example:

            input_data = {
                "query": "example query",
                "documents": [
                    Document(content="Document content", score=0.8, metadata={"date": "01 January, 2022"}),
                    Document(content="Document content", score=0.9, metadata={"date": "01 January, 2021"})
                ]
            }

            output = ranker.execute(input_data)

            # output will be a dictionary with ranked documents
        """
        config = ensure_config(config)
        self.reset_run_state()
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        ranked_documents = self.perform_llm_ranking(
            query=input_data.query,
            documents=input_data.documents,
            config=config,
            **kwargs,
        )

        return {
            "documents": ranked_documents,
        }

    def perform_llm_ranking(
        self, query: str, documents: list[Document], config: RunnableConfig, **kwargs
    ) -> list[Document]:
        """
        Performs the actual ranking of documents using the LLM.

        Args:
            query (str): The query to rank documents against.
            documents (list[Document]): The list of documents to be ranked.
            config (RunnableConfig): Configuration for the execution.
            **kwargs: Additional keyword arguments.

        Returns:
            list[Document]: A list of selected documents deemed relevant by the LLM.

        Example:

            query = "example query"
            documents = [
                Document(content="Document content", score=0.8, metadata={"date": "01 January, 2022"}),
                Document(content="Document content", score=0.9, metadata={"date": "01 January, 2021"})
            ]

            ranked_documents = ranker.perform_llm_ranking(query, documents, config)

            # ranked_documents will be a list of documents deemed relevant by the LLM
        """
        run_kwargs = kwargs | {"parent_run_id": kwargs.get("run_id")}
        inputs = [
            {"query": query, "passage": document.content} for document in documents
        ]

        prompt = prompts.Prompt(
            messages=[prompts.Message(role="user", content=self.prompt_template)]
        )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            llm_results = list(
                executor.map(
                    lambda input_data: self.call_llm(
                        input_data, prompt, config, **run_kwargs
                    ),
                    inputs,
                )
            )

        logger.debug(
            f"Node {self.name} - {self.id}: LLM processed {len(llm_results)} documents"
        )

        selected_documents = []

        for result, document in zip(llm_results, documents):
            if result == "Yes":
                selected_documents.append(document)

        logger.debug(
            f"Node {self.name} - {self.id}: LLM selected {len(selected_documents)} documents for context"
        )
        return selected_documents

    def call_llm(self, input_data, prompt, config, **run_kwargs):
        """
        Calls the LLM with the given input data and prompt.

        Args:
            input_data (dict): The input data for the LLM.
            prompt (prompts.Prompt): The prompt to be used with the LLM.
            config (RunnableConfig): Configuration for the execution.
            **run_kwargs: Additional keyword arguments.

        Returns:
            str: The result from the LLM.

        Example:

            input_data = {"query": "example query", "passage": "Document content"}
            prompt = prompts.Prompt(
                messages=[prompts.Message(role="user", content=DEFAULT_PROMPT)]
            )
            config = RunnableConfig()

            result = ranker.call_llm(input_data, prompt, config)

            # result will be the LLM's response, either 'Yes' or 'No'
        """
        llm_result = self.llm.run(
            input_data=input_data,
            prompt=prompt,
            config=config,
            run_depends=self._run_depends,
            **run_kwargs,
        )
        self._run_depends = [NodeDependency(node=self.llm).to_dict()]
        if llm_result.status != RunnableStatus.SUCCESS:
            logger.error(f"Node {self.name} - {self.id}: LLM execution failed")
            raise ValueError("LLMDocumentRanker LLM execution failed")
        return llm_result.output["content"]
