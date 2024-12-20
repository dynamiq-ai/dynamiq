from typing import Any, Callable, ClassVar, Literal

from pydantic import BaseModel, Field, PrivateAttr

from dynamiq.connections import Cohere
from dynamiq.nodes.node import Node, NodeGroup, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.types import Document
from dynamiq.utils.logger import logger


class CohereRerankerInputSchema(BaseModel):
    query: str = Field(..., description="Parameter to provide query for ranking.")
    documents: list[Document] = Field(..., description="Parameter to provide list of documents.")


class CohereReranker(Node):
    """
    A Node class for reranking documents using Cohere's reranking model.

    This ranker uses Cohere's API to rerank documents based on their relevance to a query.

    Attributes:
        group (Literal[NodeGroup.RANKERS]): The group the node belongs to.
        name (str): The name of the node.
        top_k (int): The number of top documents to return.
        model (str): The Cohere model to use for reranking.
        threshold (float): The threshold for relevance score. Default is 0.
        connection (Cohere): The Cohere connection instance.
    """

    group: Literal[NodeGroup.RANKERS] = NodeGroup.RANKERS
    name: str = "CohereReranker"
    top_k: int = 5
    model: str = "cohere/rerank-v3.5"
    threshold: float = 0
    connection: Cohere
    input_schema: ClassVar[type[CohereRerankerInputSchema]] = CohereRerankerInputSchema
    _rerank: Callable = PrivateAttr()

    def __init__(self, **kwargs):
        """Initialize the CohereReranker instance."""
        super().__init__(**kwargs)

        from litellm import rerank

        self._rerank = rerank

    def execute(self, input_data: CohereRerankerInputSchema, config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        """
        Execute the document reranking process.

        Args:
            input_data (CohereRerankerInputSchema): The input data containing documents and query.
            config (RunnableConfig, optional): Configuration for the runnable.
            **kwargs: Additional keyword arguments.

        Returns:
            dict[str, Any]: A dictionary containing the reranked documents.
        """
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")

        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        query = input_data.query
        documents = input_data.documents

        if not documents:
            logger.warning(f"Node {self.name} - {self.id}: No documents provided for reranking")
            return {"documents": []}

        document_texts = [doc.content for doc in documents]

        logger.debug(f"Node {self.name} - {self.id}: Reranking {len(documents)} documents")

        response = self._rerank(model=self.model, query=query, documents=document_texts, top_n=self.top_k)

        reranked_documents = []
        for result in response.results:
            doc = documents[result.get("index")]
            doc.score = result.get("relevance_score")
            if doc.score > self.threshold:
                reranked_documents.append(doc)

        logger.debug(f"Node {self.name} - {self.id}: Successfully reranked {len(reranked_documents)} documents")

        return {"documents": reranked_documents[: self.top_k]}
