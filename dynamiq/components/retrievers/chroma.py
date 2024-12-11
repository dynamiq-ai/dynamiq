from typing import Any

from dynamiq.storages.vector import ChromaVectorStore
from dynamiq.types import Document


class ChromaDocumentRetriever:
    """
    Document Retriever using Chroma.
    """

    def __init__(
        self,
        *,
        vector_store: ChromaVectorStore,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
    ):
        """
        Initializes a component for retrieving documents from a Chroma vector store with optional filtering.

        Args:
            vector_store (ChromaVectorStore): An instance of ChromaVectorStore to interface with Chroma vectors.
            filters (Optional[dict[str, Any]]): Filters to apply for retrieving specific documents. Defaults to None.
            top_k (int): The maximum number of documents to return. Defaults to 10.

        Raises:
            ValueError: If the `vector_store` is not an instance of `ChromaVectorStore`.

        This initializer checks if the `vector_store` provided is an instance of the expected `ChromaVectorStore`
        class, sets up filtering conditions if any, and defines how many top results to retrieve in document queries.
        """
        if not isinstance(vector_store, ChromaVectorStore):
            msg = "document_store must be an instance of ChromaVectorStore"
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
    ) -> dict[str, list[Document]]:
        """
        Retrieves documents from the ChromaVectorStore that are similar to the provided query embedding.

        Args:
            query_embedding (List[float]): The embedding vector of the query for which similar documents are to be
            retrieved.
            exclude_document_embeddings (bool, optional): Specifies whether to exclude the embeddings of the retrieved
            documents from the output.
            top_k (int, optional): The maximum number of documents to return. Defaults to None.
            filters (Optional[dict[str, Any]]): Filters to apply for retrieving specific documents. Defaults to None.

        Returns:
            List[Document]: A list of Document instances sorted by their relevance to the query_embedding.
        """
        query_embeddings = [query_embedding]
        top_k = top_k or self.top_k
        filters = filters or self.filters

        docs = self.vector_store.search_embeddings(
            query_embeddings=query_embeddings,
            filters=filters,
            top_k=top_k,
        )[0]

        if exclude_document_embeddings:
            for doc in docs:
                doc.embedding = None
        return {"documents": docs}
