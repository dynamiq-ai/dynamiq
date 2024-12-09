from typing import Any

from dynamiq.storages.vector import PineconeVectorStore
from dynamiq.types import Document
from dynamiq.utils.logger import logger


class PineconeDocumentRetriever:
    """
    Document Retriever using Pinecone.
    """

    def __init__(
        self,
        *,
        vector_store: PineconeVectorStore,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
    ):
        """
        Initializes a component for retrieving documents from a Pinecone vector store with optional filtering.

        Args:
            vector_store (PineconeVectorStore): An instance of PineconeVectorStore to interface with Pinecone vectors.
            filters (Optional[dict[str, Any]]): Filters to apply for retrieving specific documents. Defaults to None.
            top_k (int): The maximum number of documents to return. Defaults to 10.

        Raises:
            ValueError: If the `vector_store` is not an instance of `PineconeVectorStore`.

        This initializer checks if the `vector_store` provided is an instance of the expected `PineconeVectorStore`
        class, sets up filtering conditions if any, and defines how many top results to retrieve in document queries.
        """
        if not isinstance(vector_store, PineconeVectorStore):
            msg = "document_store must be an instance of PineconeVectorStore"
            raise ValueError(msg)

        self.vector_store = vector_store
        self.filters = filters or {}
        self.top_k = top_k

    def run(
        self,
        query_embedding: list[float],
        exclude_document_embeddings: bool = True,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        content_key: str | None = None,
    ) -> dict[str, list[Document]]:
        """
        Retrieves documents from the PineconeDocumentStore that are similar to the provided query embedding.

        Args:
            query_embedding (List[float]): The embedding vector of the query for which similar documents are to be
            retrieved.
            exclude_document_embeddings (bool, optional): Specifies whether to exclude the embeddings of the retrieved
            documents from the output.
            top_k (int, optional): The maximum number of documents to return. Defaults to None.
            filters (Optional[dict[str, Any]]): Filters to apply for retrieving specific documents. Defaults to None.
            content_key (Optional[str]): The field used to store content in the storage.

        Returns:
            List[Document]: A list of Document instances sorted by their relevance to the query_embedding.
        """
        top_k = top_k or self.top_k
        filters = filters or self.filters

        docs = self.vector_store._embedding_retrieval(
            query_embedding=query_embedding,
            filters=filters,
            top_k=top_k,
            exclude_document_embeddings=exclude_document_embeddings,
            content_key=content_key,
        )
        logger.debug(f"Retrieved {len(docs)} documents from Pinecone Vector Store.")

        return {"documents": docs}
