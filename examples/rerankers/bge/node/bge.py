from typing import Any, Literal

from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.node import Node, NodeGroup, ensure_config
from dynamiq.runnables import RunnableConfig
from examples.rerankers.bge.component.document import DocumentRanker as DocumentRankerComponent


class BGEDocumentRanker(Node):
    """
    A document ranker node using the BGE (BAAI/bge-reranker) model.

    This node ranks documents based on their relevance to a given query using the BGE reranker model.

    Attributes:
        group (Literal[NodeGroup.RANKERS]): The group of the node, set to RANKERS.
        name (str): The name of the node, set to "DocumentRanker".
        model_name (str): The name of the BGE model to use, default is "BAAI/bge-reranker-v2-m3".
        threshold (float): The threshold score for document ranking, default is 0.0.
        top_k (int): The number of top-ranked documents to return, default is 5.
        ranker (DocumentRankerComponent | None): The document ranker component, initialized in
            init_components.
    """

    group: Literal[NodeGroup.RANKERS] = NodeGroup.RANKERS
    name: str = "DocumentRanker"
    model_name: str = "BAAI/bge-reranker-v2-m3"
    threshold: float = 0.0
    top_k: int = 5
    ranker: DocumentRankerComponent | None = None

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {"ranker": True}

    def init_components(
        self, connection_manager: ConnectionManager = ConnectionManager()
    ) -> None:
        """
        Initialize the document ranker component.

        Args:
            connection_manager (ConnectionManager): The connection manager to use, default is a new
                instance.
        """
        super().init_components(connection_manager)
        if self.ranker is None:
            self.ranker = DocumentRankerComponent(
                model_name=self.model_name, threshold=self.threshold
            )

    def execute(
        self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs
    ) -> dict[str, Any]:
        """
        Execute the document ranking process.

        This method ranks the input documents based on their relevance to the given query using the
        BGE reranker model.

        Args:
            input_data (dict[str, Any]): A dictionary containing the 'query' and 'documents' to rank.
            config (RunnableConfig, optional): Configuration for the execution. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            dict[str, Any]: A dictionary containing the original query and the ranked documents.
        """
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        ranked_documents = self.ranker.run(
            query=input_data["query"], documents=input_data["documents"]
        )

        return {
            "query": input_data["query"],
            "documents": ranked_documents,
        }
