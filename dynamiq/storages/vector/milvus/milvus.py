from copy import copy
from typing import TYPE_CHECKING, Any, Optional

from pymilvus import Collection, DataType

from dynamiq.connections import Milvus
from dynamiq.storages.vector.base import BaseWriterVectorStoreParams
from dynamiq.storages.vector.milvus.filters import _normalize_filters
from dynamiq.types import Document
from dynamiq.utils.logger import logger

if TYPE_CHECKING:
    from pymilvus import MilvusClient


class MilvusWriterVectorStoreParams(BaseWriterVectorStoreParams):
    dimension: int = 1536
    metric: str = "COSINE"
    limit: int = 5


class MilvusVectorStore:
    def __init__(
        self,
        connection: Milvus | None = None,
        client: Optional["MilvusClient"] = None,
        dimension: int = 1536,
        collection_name: str = "default",
        partition_name: str = "default",
        metric: str = "COSINE",
        limit: int = 5,
        timeout: int | None = None,
        auto_id: bool = False,
        create_if_not_exist: bool = False,
        **index_creation_kwargs,
    ):
        self.client = client
        if self.client is None:
            if connection is None:
                connection = Milvus()
            self.client = connection.connect()
        self.partition_name = partition_name
        self.dimension = dimension
        self.metric = metric
        self.limit = limit
        self.collection_name = collection_name
        self.timeout = timeout
        self._dummy_vector = [-10.0] * dimension
        self.auto_id = auto_id
        self.create_if_not_exist = create_if_not_exist
        self.index_creation_kwargs = index_creation_kwargs
        self.collection = self.create_collection()
        logger.debug(
            f"MilvusVectorStore initialized with collection {self.collection_name} and partition "
            f"{self.partition_name}."
        )

    def create_collection(self):
        """
        Create or connect to an existing Milvus collection.

        """
        available_collections = self.client.list_collections()
        if self.collection_name not in available_collections:
            if self.create_if_not_exist:
                logger.debug(f"Collection {self.collection_name} does not exist. Creating a new collection.")
                schema = self.client.create_schema(
                    auto_id=self.auto_id,
                )

                schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=512, is_primary=True)
                schema.add_field(
                    field_name="vector", datatype=DataType.FLOAT_VECTOR, metric_type=self.metric, dim=self.dimension
                )
                schema.add_field(field_name="metadata", datatype=DataType.JSON)

                collection = Collection(
                    name=self.collection_name,
                    schema=schema,
                )
                collection.create_partition(
                    partition_name=self.partition_name,
                )
                collection.create_index(
                    field_name="vector",
                    index_params={"index_type": "IVF_FLAT", "metric_type": self.metric, "params": {"nlist": 1024}},
                )
                self.client.load_collection(collection_name=self.collection_name)
                return collection
            else:
                raise ValueError(
                    f"Collection {self.collection_name} does not exist."
                    f"'create_if_not_exist' must be set to True if you want to create a new collection automatically."
                )
        else:
            logger.debug(f"Collection {self.collection_name} already exists and can be used.")
            return Collection(name=self.collection_name)

    def write_documents(self, documents: list[Document]) -> int:
        """
        Write documents to the Milvus vector store.

        Args:
            documents (list[Document]): List of Document objects to write.

        Returns:
            int: Number of documents successfully written.

        Raises:
            ValueError: If documents are not of type Document.
        """
        if len(documents) > 0 and not isinstance(documents[0], Document):
            msg = "param 'documents' must contain a list of objects of type Document"
            raise ValueError(msg)

        documents_for_milvus = self._convert_documents_to_milvus_format(documents)

        result = self.client.upsert(
            collection_name=self.collection_name, data=documents_for_milvus, partition_name=self.partition_name
        )

        written_docs = result["upsert_count"]
        return written_docs

    def _convert_documents_to_milvus_format(self, documents: list[Document]) -> list[dict[str, Any]]:
        """
        Convert Document objects to Milvus-compatible format.

        Args:
            documents (list[Document]): List of Document objects to convert.

        Returns:
            list[dict[str, Any]]: List of documents in Milvus-compatible format.
        """
        documents_for_milvus = []
        for document in documents:
            embedding = copy(document.embedding)
            if embedding is None:
                logger.warning(f"Document {document.id} has no embedding. A dummy embedding will be used.")
                embedding = self._dummy_vector
            doc_for_milvus = {
                "id": document.id,
                "vector": embedding,
                "metadata": dict(document.metadata),
            }

            if document.content is not None:
                doc_for_milvus["metadata"]["content"] = document.content

            documents_for_milvus.append(doc_for_milvus)
        return documents_for_milvus

    def list_documents(self, include_embeddings: bool = False) -> list[Document]:
        """
        List documents in the Milvus vector store.

        Args:
            include_embeddings (bool): Whether to include embeddings in the results. Defaults to False.

        Returns:
            list[Document]: List of Document objects retrieved.
        """

        response = self.collection.query(
            expr="", limit=self.limit, output_fields=["*"], partition_names=[self.partition_name]
        )

        documents = []
        for milvus_doc in response:
            content = milvus_doc["metadata"].pop("content", None)

            embedding = None
            if include_embeddings and milvus_doc["vector"] != self._dummy_vector:
                embedding = milvus_doc["vector"]

            doc = Document(
                id=milvus_doc["id"],
                content=content,
                metadata=milvus_doc["metadata"],
                embedding=embedding,
                score=None,
            )
            documents.append(doc)

        return documents

    def count_documents(self) -> int:
        """
        Count the number of documents in the store.

        Returns:
            int: The number of documents in the store.
        """
        try:
            count = self.client.get_collection_stats(self.collection_name)["row_count"]
        except KeyError:
            count = 0
        return count

    def delete_documents(self, document_ids: list[str] | str | None = None, delete_all: bool = False) -> None:
        """
        Delete documents from the Milvus vector store.

        Args:
            document_ids (list[str]): List of document IDs to delete. Defaults to None.
            delete_all (bool): If True, delete all documents. Defaults to False.
        """
        if delete_all and self.client.has_collection(collection_name=self.collection_name) is True:
            self.collection.delete(expr="id != ''", partition_name=self.partition_name)
        else:
            if not document_ids:
                logger.warning("No document IDs provided. No documents will be deleted.")
            else:
                self.client.delete(
                    ids=document_ids, collection_name=self.collection_name, partition_name=self.partition_name
                )

    def delete_collection(self):
        """Delete the entire collection."""
        self.client.drop_collection(collection_name=self.collection_name)

    def delete_documents_by_filters(self, filters: dict[str, Any], top_k: int = 1000) -> None:
        """
        Delete documents from the Milvus vector store using filters.

        Args:
            filters (dict[str, Any]): Filters to select documents to delete.
            top_k (int): Maximum number of documents to retrieve for deletion. Defaults to 1000.
        """
        filters = _normalize_filters(filters)
        deletion_info = self.collection.delete(expr=filters, partition_name=self.partition_name)
        logger.info("Deleted %d documents from Milvus vector store.", deletion_info.delete_count)

    def _embedding_retrieval(
        self,
        query_embedding: list[float],
        partition_name: str | None = None,
        *,
        output_fields: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        limit: int = 10,
    ) -> list[Document]:
        """
        Retrieve documents similar to the given query embedding.

        Args:
            query_embedding (list[float]): The query embedding vector.
            filters (dict[str, Any] | None): Filters for the query. Defaults to None.
            limit (int): Maximum number of documents to retrieve. Defaults to 10.

        Returns:
            list[Document]: List of retrieved Document objects.

        Raises:
            ValueError: If query_embedding is empty or filter format is incorrect.
        """
        if not query_embedding:
            msg = "query_embedding must be a non-empty list of floats"
            raise ValueError(msg)

        filters = _normalize_filters(filters) if filters else ""

        result = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=limit,
            filter=filters,
            search_params={"metric_type": self.metric, "params": {}},
            partition_names=[partition_name or self.partition_name],
            output_fields=output_fields or ["*"],
        )

        return self._convert_query_result_to_documents(result)

    def _convert_query_result_to_documents(self, query_result: list[list[dict[str, Any]]]) -> list[Document]:
        """
        Convert Milvus query results to Document objects.

        Args:
            query_result (dict[str, Any]): The query result from Milvus.

        Returns:
            list[Document]: List of Document objects created from the query result.
        """
        milvus_docs = query_result[0]
        documents = []
        for milvus_doc in milvus_docs:
            content = milvus_doc["entity"]["metadata"].pop("content", None)

            embedding = None
            if milvus_doc["entity"]["vector"] != self._dummy_vector:
                embedding = milvus_doc["entity"]["vector"]

            doc = Document(
                id=milvus_doc["id"],
                content=content,
                metadata=milvus_doc["entity"]["metadata"],
                embedding=embedding,
                score=milvus_doc["distance"],
            )
            documents.append(doc)

        return documents
