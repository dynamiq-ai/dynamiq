from typing import TYPE_CHECKING, Any, Optional

from pydantic.types import PositiveInt
from pymilvus import AnnSearchRequest, DataType, Function, FunctionType, RRFRanker

from dynamiq.connections import Milvus
from dynamiq.storages.vector.base import BaseVectorStore, BaseVectorStoreParams, BaseWriterVectorStoreParams
from dynamiq.storages.vector.milvus.filter import Filter
from dynamiq.types import Document
from dynamiq.utils.logger import logger

if TYPE_CHECKING:
    from pymilvus import MilvusClient


class MilvusVectorStoreParams(BaseVectorStoreParams):
    embedding_key: str = "embedding"


class MilvusWriterVectorStoreParams(MilvusVectorStoreParams, BaseWriterVectorStoreParams):
    dimension: PositiveInt = 1536


class MilvusVectorStore(BaseVectorStore):
    """
    Vector store using Milvus.

    This class can be used with Zilliz Cloud Services or self-hosted instances.

    """

    def __init__(
        self,
        connection: Milvus | None = None,
        client: Optional["MilvusClient"] = None,
        index_name: str = "default",
        metric_type: str = "COSINE",
        index_type: str = "AUTOINDEX",
        dimension: int = 1536,
        create_if_not_exist: bool = False,
        content_key: str = "content",
        embedding_key: str = "embedding",
    ):
        self.client = client
        if self.client is None:
            connection = connection or Milvus()
            self.client = connection.connect()
        self.index_name = index_name
        self.metric_type = metric_type
        self.index_type = index_type
        self.content_key = content_key
        self.embedding_key = embedding_key
        self.dimension = dimension
        self.create_if_not_exist = create_if_not_exist
        self.schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
        self.schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=65_535)
        self.schema.add_field(
            field_name=self.content_key, datatype=DataType.VARCHAR, max_length=65_535, enable_analyzer=True
        )
        self.schema.add_field(field_name=self.embedding_key, datatype=DataType.FLOAT_VECTOR, dim=self.dimension)
        self.schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)

        bm25_function = Function(
            name="text_bm25_emb",
            input_field_names=[self.content_key],
            output_field_names=["sparse"],
            function_type=FunctionType.BM25,
        )
        self.schema.add_function(bm25_function)

        self.index_params = self.client.prepare_index_params()
        self.index_params.add_index(field_name="id")
        self.index_params.add_index(
            field_name=self.embedding_key, index_type=self.index_type, metric_type=self.metric_type
        )
        self.index_params.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")

        if not self.client.has_collection(self.index_name):
            if self.create_if_not_exist:
                logger.info(f"Collection {self.index_name} does not exist. Creating a new collection.")
                self.client.create_collection(
                    collection_name=self.index_name, schema=self.schema, index_params=self.index_params
                )
            else:
                raise ValueError(
                    f"Collection {self.index_name} does not exist. Set 'create_if_not_exist' to True to create it."
                )
        else:
            logger.info(f"Collection {self.index_name} already exists. Skipping creation.")

        self.client.load_collection(self.index_name)

    def count_documents(self) -> int:
        """
        Get the number of documents in the collection.

        Returns:
            int: The number of documents in the collection.
        """
        return self.client.get_collection_stats(self.index_name)["row_count"]

    def write_documents(
        self, documents: list[Document], content_key: str | None = None, embedding_key: str | None = None
    ) -> int:
        """
        Write (or overwrite) documents into the Milvus store.

        This method processes a list of Document objects and writes them into the vector store.

        Args:
            documents (List[Document]): A list of Document objects to be written into the document store.
            content_key (Optional[str]): The field used to store content in the storage.
            embedding_key (Optional[str]): The field used to store vector in the storage.

        Raises:
            ValueError: If an item in the documents list is not an instance of the Document class.

        Returns:
            int: The number of documents successfully written to the document store.
        """
        data_to_upsert = []
        content_key = content_key or self.content_key
        embedding_key = embedding_key or self.embedding_key
        for doc in documents:
            if not isinstance(doc, Document):
                raise ValueError("All items in 'documents' must be of type Document.")

            document_data = {
                "id": doc.id,
                embedding_key: doc.embedding,
                content_key: doc.content,
            }

            if doc.metadata:
                document_data.update(doc.metadata)

            data_to_upsert.append(document_data)

        response = self.client.upsert(
            collection_name=self.index_name,
            data=data_to_upsert,
        )
        return response["upsert_count"]

    def delete_documents(self, document_ids: list[str] | None = None, delete_all: bool = False) -> None:
        """
        Delete documents from the Milvus vector store based on their IDs.

        Args:
            document_ids (List[str]): A list containing the IDs of documents to be deleted from the store.
            delete_all (bool): A flag to delete all documents from the store. Defaults to False.

        Raises:
            ValueError: If neither document_ids nor delete_all is provided.
        """
        if delete_all:
            self.client.drop_collection(collection_name=self.index_name)
            self.client.create_collection(
                collection_name=self.index_name, schema=self.schema, index_params=self.index_params
            )
            self.client.load_collection(self.index_name)
            logger.info(f"All documents in the collection {self.index_name} have been deleted.")
        elif document_ids:
            response = self.client.delete(collection_name=self.index_name, ids=document_ids)
            logger.info(f"Deleted {len(response)} documents from collection {self.index_name}.")
        else:
            raise ValueError("Either `document_ids` or `delete_all` must be provided.")

    def delete_documents_by_filters(self, filters: dict[str, Any]) -> None:
        """
        Delete documents based on filters.

        Args:
            filters (Dict[str, Any]): Filter criteria for deleting documents.
        """
        if not filters:
            raise ValueError("Filters must be provided to delete documents.")

        filter_expression = Filter(filters).build_filter_expression()

        delete_result = self.client.delete(collection_name=self.index_name, filter=filter_expression)

        logger.info(f"Deleted {len(delete_result)} entities from collection {self.index_name} based on filters.")

    def list_documents(
        self, limit: int = 1000, content_key: str | None = None, embedding_key: str | None = None
    ) -> list[Document]:
        """
        List all documents in the collection up to a specified limit.

        Args:
            limit (int): Maximum number of documents to retrieve. Defaults to 1000.
            content_key (Optional[str]): The field used to store content in the storage.
            embedding_key (Optional[str]): The field used to store vector in the storage.

        Returns:
            List[Document]: A list of Document instances representing all documents in the collection.
        """
        if not self.client.has_collection(self.index_name):
            raise ValueError(f"Collection '{self.index_name}' does not exist.")

        result = self.client.query(collection_name=self.index_name, filter="", output_fields=["*"], limit=limit)

        return self._get_result_to_documents(result, content_key=content_key, embedding_key=embedding_key)

    def _embedding_retrieval(
        self,
        query_embeddings: list[list[float]],
        top_k: int,
        filters: dict[str, Any] | None = None,
        content_key: str | None = None,
        embedding_key: str | None = None,
        return_embeddings: bool = False,
    ) -> list[Document]:
        """
        Perform vector search on the stored documents using query embeddings.

        Args:
            query_embeddings (list[list[float]]): A list of embeddings to use as queries.
            top_k (int): The maximum number of documents to retrieve.
            filters (dict[str, Any] | None): A dictionary of filters to apply to the search. Defaults to None.
            content_key (Optional[str]): The field used to store content in the storage.
            embedding_key (Optional[str]): The field used to store vector in the storage.
            return_embeddings (bool): Whether to return the embeddings of the retrieved documents.

        Returns:
            List[Document]: A list of Document objects containing the retrieved documents.
        """
        search_params = {"metric_type": self.metric_type, "params": {}}

        filter_expression = Filter(filters).build_filter_expression() if filters else ""

        results = self.client.search(
            collection_name=self.index_name,
            data=query_embeddings,
            limit=top_k,
            filter=filter_expression,
            output_fields=["*"],
            anns_field=embedding_key or self.embedding_key,
            search_params=search_params,
        )

        return self._convert_query_result_to_documents(
            results[0], content_key=content_key, embedding_key=embedding_key, return_embeddings=return_embeddings
        )

    def _convert_query_result_to_documents(
        self,
        result: list[dict[str, Any]],
        content_key: str | None = None,
        embedding_key: str | None = None,
        return_embeddings: bool = False,
    ) -> list[Document]:
        """
        Convert Milvus search results to Document objects.

        Args:
            result (List[Dict[str, Any]]): The result from a Milvus search operation.
            content_key (Optional[str]): The field used to store content in the storage.
            embedding_key (Optional[str]): The field used to store vector in the storage.
            return_embeddings (bool): Whether to return the embeddings of the retrieved documents.

        Returns:
            List[Document]: A list of Document instances created from the Milvus search result.
        """
        documents = []
        content_key = content_key or self.content_key
        embedding_key = embedding_key or self.embedding_key
        for hit in result:
            entity = hit.get("entity", {})
            content = entity.get(content_key, "")
            embedding = entity.get(embedding_key, [])
            metadata = {k: v for k, v in entity.items() if k not in (content_key, embedding_key)}

            doc = Document(
                id=str(hit.get("id", "")),
                content=content,
                metadata=metadata,
                score=hit.get("distance", None),
            )
            if return_embeddings:
                doc.embedding = embedding

            documents.append(doc)

        return documents

    def filter_documents(
        self, filters: dict[str, Any] | None = None, content_key: str | None = None, embedding_key: str | None = None
    ) -> list[Document]:
        """
        Retrieve documents that match the provided filters.

        Args:
            filters (Dict[str, Any] | None): The filters to apply to the document list.
            content_key (Optional[str]): The field used to store content in the storage.
            embedding_key (Optional[str]): The field used to store vector in the storage.

        Returns:
            list[Document]: A list of Document instances that match the given filters.

        Raises:
            ValueError: If no filters are provided.
        """
        if not filters:
            raise ValueError("No filters provided. No documents will be retrieved with filters.")

        filter_expression = Filter(filters).build_filter_expression()

        result = self.client.query(
            collection_name=self.index_name,
            filter=filter_expression,
            output_fields=["*"],
        )
        return self._get_result_to_documents(result, content_key=content_key, embedding_key=embedding_key)

    def _get_result_to_documents(
        self, result: list[dict[str, Any]], content_key: str | None = None, embedding_key: str | None = None
    ) -> list[Document]:
        """
        Convert Milvus query result into Documents.

        Args:
            result (List[Dict[str, Any]]): The result from a Milvus query operation.
            content_key (Optional[str]): The field used to store content in the storage.
            embedding_key (Optional[str]): The field used to store vector in the storage.

        Returns:
            List[Document]: A list containing Document objects created from the Milvus query result.
        """
        documents = []
        content_key = content_key or self.content_key
        embedding_key = embedding_key or self.embedding_key
        for entry in result:
            document_dict: dict[str, Any] = {
                "id": str(entry.get("id", "")),
                "content": entry.get(content_key, ""),
                "embedding": entry.get(embedding_key, []),
            }
            metadata = {k: v for k, v in entry.items() if k not in ("id", content_key, embedding_key)}

            if metadata:
                document_dict["metadata"] = metadata

            try:
                documents.append(Document(**document_dict))
            except Exception as e:
                logger.error(f"Error creating Document: {e}, data: {document_dict}")

        return documents

    def _hybrid_retrieval(
        self,
        query: str,
        query_embeddings: list[list[float]],
        top_k: int,
        top_k_dense: int | None = None,
        top_k_sparse: int | None = None,
        content_key: str | None = None,
        embedding_key: str | None = None,
        return_embeddings: bool = False,
        drop_ratio_build: float = 0.0,
    ) -> list[Document]:
        """
        Perform a hybrid search using both dense (vector-based) and sparse (text-based) retrieval techniques.

        Args:
            query (str): The textual query used for sparse search (BM25).
            query_embeddings (list[list[float]]): A list of embeddings representing the query for dense search.
            top_k (int): The maximum number of documents to retrieve with hybrid search.
            top_k_dense (int | None, optional): The number of top results to retrieve from the dense search.
                If None, defaults to `top_k`.
            top_k_sparse (int | None, optional): The number of top results to retrieve from the sparse search.
                If None, defaults to `top_k`.
            content_key (Optional[str]): The field used to store content in the storage.
            embedding_key (Optional[str]): The field used to store vector in the storage.
            return_embeddings (bool): Whether to return the embeddings of the retrieved documents.
            drop_ratio_build (float): The ratio of small vector values to be dropped during indexing during text search.

        Returns:
            List[Document]: A list of Document objects containing the retrieved documents.
        """

        search_param_1 = {
            "data": query_embeddings,
            "anns_field": embedding_key or self.embedding_key,
            "param": {
                "metric_type": self.metric_type,
            },
            "limit": top_k_dense or top_k,
        }
        request_1 = AnnSearchRequest(**search_param_1)

        search_param_2 = {
            "data": [query],
            "anns_field": "sparse",
            "param": {"metric_type": "BM25", "params": {"drop_ratio_build": drop_ratio_build}},
            "limit": top_k_sparse or top_k,
        }
        request_2 = AnnSearchRequest(**search_param_2)

        ranker = RRFRanker()
        results = self.client.hybrid_search(
            collection_name=self.index_name,
            output_fields=["*"],
            reqs=[request_1, request_2],
            ranker=ranker,
            limit=top_k,
        )

        return self._convert_query_result_to_documents(
            results[0], content_key=content_key, embedding_key=embedding_key, return_embeddings=return_embeddings
        )
