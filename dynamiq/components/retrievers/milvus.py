from typing import Any

from dynamiq.storages.vector import MilvusVectorStore
from dynamiq.types import Document


class MilvusDocumentRetriever:
    """
    Document Retriever using Milvus.
    """

    def __init__(
        self,
        *,
        vector_store: MilvusVectorStore,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
    ):
        """
        Initializes a component for retrieving documents from a Milvus vector store with optional filtering.

        Args:
            vector_store (MilvusVectorStore): An instance of MilvusVectorStore to interface with Milvus vectors.
            filters (Optional[dict[str, Any]]): Filters to apply for retrieving specific documents. Defaults to None.
            top_k (int): The maximum number of documents to return. Defaults to 10.

        Raises:
            ValueError: If the `vector_store` is not an instance of `MilvusVectorStore`.
        """
        if not isinstance(vector_store, MilvusVectorStore):
            raise ValueError("vector_store must be an instance of MilvusVectorStore")

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
        embedding_key: str | None = None,
    ) -> dict[str, list[Document]]:
        """
        Retrieves documents from the MilvusVectorStore that are similar to the provided query embedding.

        Args:
            query_embedding (List[float]): The embedding vector of the query for which similar documents are to be
            retrieved.
            exclude_document_embeddings (bool, optional): Specifies whether to exclude the embeddings of the retrieved
            documents from the output.
            top_k (int, optional): The maximum number of documents to return. Defaults to None.
            filters (Optional[dict[str, Any]]): Filters to apply for retrieving specific documents. Defaults to None.
            content_key (Optional[str]): The field used to store content in the storage.
            embedding_key (Optional[str]): The field used to store vector in the storage.

        Returns:
            Dict[str, List[Document]]: A dictionary containing a list of Document instances sorted by their relevance
            to the query_embedding.
        """
        query_embeddings = [query_embedding]
        top_k = top_k or self.top_k
        filters = filters or self.filters

        docs = self.vector_store.search_embeddings(
            query_embeddings=query_embeddings,
            filters=filters,
            top_k=top_k,
            content_key=content_key,
            embedding_key=embedding_key,
        )

        # Optionally exclude embeddings from the retrieved documents
        if exclude_document_embeddings:
            for doc in docs:
                doc.embedding = None
        return {"documents": docs}
