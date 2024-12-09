import enum
from copy import copy
from functools import partial
from typing import TYPE_CHECKING, Any, Optional

from pydantic import Field

from dynamiq.connections import Pinecone
from dynamiq.storages.vector.base import BaseVectorStoreParams, BaseWriterVectorStoreParams
from dynamiq.storages.vector.pinecone.filters import _normalize_filters
from dynamiq.storages.vector.utils import create_file_id_filter
from dynamiq.types import Document
from dynamiq.utils.env import get_env_var
from dynamiq.utils.logger import logger

if TYPE_CHECKING:
    from pinecone import Pinecone as PineconeClient


class PineconeIndexType(str, enum.Enum):
    """
    This enum defines various index types for different Pinecone deployments.
    """

    SERVERLESS = "serverless"
    POD = "pod"


class PineconeVectorStoreParams(BaseVectorStoreParams):
    namespace: str = "default"


class PineconeWriterVectorStoreParams(PineconeVectorStoreParams, BaseWriterVectorStoreParams):
    batch_size: int = 100
    dimension: int = 1536
    metric: str = "cosine"
    index_type: PineconeIndexType | None = None
    cloud: str | None = Field(default_factory=partial(get_env_var, "PINECONE_CLOUD"))
    region: str | None = Field(default_factory=partial(get_env_var, "PINECONE_REGION"))
    environment: str | None = Field(default_factory=partial(get_env_var, "PINECONE_ENVIRONMENT"))
    pod_type: str | None = Field(default_factory=partial(get_env_var, "PINECONE_POD_TYPE"))
    pods: int = 1


class PineconeVectorStore:
    """Vector store using Pinecone."""

    def __init__(
        self,
        connection: Pinecone | None = None,
        client: Optional["PineconeClient"] = None,
        index_name: str = "default",
        namespace: str = "default",
        batch_size: int = 100,
        dimension: int = 1536,
        metric: str = "cosine",
        create_if_not_exist: bool = False,
        index_type: PineconeIndexType | None = None,
        cloud: str | None = None,
        region: str | None = None,
        environment: str | None = None,
        pod_type: str | None = None,
        pods: int = 1,
        content_key: str = "content",
        **index_creation_kwargs,
    ):
        """
        Initialize a PineconeVectorStore instance.

        Args:
            connection (Optional[Pinecone]): Pinecone connection instance. Defaults to None.
            client (Optional[PineconeClient]): Pinecone client instance. Defaults to None.
            index_name (str): Name of the Pinecone index. Defaults to None.
            namespace (str): Namespace for the index. Defaults to 'default'.
            batch_size (int): Size of batches for operations. Defaults to 100.
            dimension (int): Number of dimensions for vectors. Defaults to 1536.
            metric (str): Metric for calculating vector similarity. Defaults to 'cosine'.
            content_key (Optional[str]): The field used to store content in the storage. Defaults to 'content'.
            **index_creation_kwargs: Additional arguments for index creation.
        """
        self.client = client
        if self.client is None:
            if connection is None:
                connection = Pinecone()
            self.client = connection.connect()

        self.index_name = index_name
        self.namespace = namespace
        self.index_type = index_type
        self.content_key = content_key

        self.create_if_not_exist = create_if_not_exist

        self.batch_size = batch_size
        self.metric = metric
        self.dimension = dimension
        self.cloud = cloud
        self.region = region
        self.environment = environment
        self.pod_type = pod_type
        self.pods = pods

        self.index_creation_kwargs = index_creation_kwargs

        self._spec = self._get_spec()
        self._dummy_vector = [-10.0] * dimension
        self._index = self.connect_to_index()
        logger.debug(f"PineconeVectorStore initialized with index {self.index_name} and namespace {self.namespace}.")

    def _get_spec(self):
        """
        Returns the serverless or pod specification for the Pinecone service.

        Returns:
            ServerlessSpec | PodSpec | None: The serverless or pod specification.
        """
        if self.index_type == PineconeIndexType.SERVERLESS:
            return self.serverless_spec
        elif self.index_type == PineconeIndexType.POD:
            return self.pod_spec

    @property
    def serverless_spec(self):
        """
        Returns the serverless specification for the Pinecone service.

        Returns:
            ServerlessSpec: The serverless specification.
        """
        # Import in runtime to save memory
        from pinecone import ServerlessSpec

        if self.cloud is None or self.region is None:
            raise ValueError("'cloud' and 'region' must be specified for 'serverless' index")
        return ServerlessSpec(cloud=self.cloud, region=self.region)

    @property
    def pod_spec(self):
        """
        Returns the pod specification for the Pinecone service.

        Returns:
            PodSpec: The pod specification.
        """
        # Import in runtime to save memory
        from pinecone import PodSpec

        if self.environment is None or self.pod_type is None:
            raise ValueError("'environment' and 'pod_type' must be specified for 'pod' index")

        return PodSpec(environment=self.environment, pod_type=self.pod_type, pods=self.pods)

    def connect_to_index(self):
        """
        Create or connect to an existing Pinecone index.

        Returns:
            The initialized Pinecone index object.
        """
        available_indexes = self.client.list_indexes().index_list["indexes"]
        indexes_names = [index["name"] for index in available_indexes]

        if self.index_name not in indexes_names:
            if self.create_if_not_exist and self.index_type is not None:
                logger.debug(f"Index {self.index_name} does not exist. Creating a new index.")
                self.client.create_index(
                    name=self.index_name,
                    spec=self._spec,
                    dimension=self.dimension,
                    metric=self.metric,
                    **self.index_creation_kwargs,
                )
                return self.client.Index(name=self.index_name)
            else:
                raise ValueError(
                    f"Index {self.index_name} does not exist."
                    f"'create_if_not_exist' must be set to True and 'index_type' must be specified."
                )
        else:
            logger.debug(f"Index {self.index_name} already exists. Connecting to it.")
            return self.client.Index(name=self.index_name)

    def _set_dimension(self, dimension: int):
        """
        Set the dimension for the index, with a warning if it differs from the actual dimension.

        Args:
            dimension (int): The desired dimension.

        Returns:
            int: The actual dimension of the index.
        """
        actual_dimension = self._index.describe_index_stats().get("dimension")
        if actual_dimension and actual_dimension != dimension:
            logger.warning(
                f"Dimension of index {self.index_name} is {actual_dimension}, but {dimension} was specified. "
                "The specified dimension will be ignored. "
                "If you need an index with a different dimension, please create a new one."
            )
        return actual_dimension or dimension

    def delete_index(self):
        """Delete the entire index."""
        self._index.delete(delete_all=True, namespace=self.namespace)
        self.client.delete_index(self.index_name)

    def delete_documents(self, document_ids: list[str] | None = None, delete_all: bool = False) -> None:
        """
        Delete documents from the Pinecone vector store.

        Args:
            document_ids (list[str]): List of document IDs to delete. Defaults to None.
            delete_all (bool): If True, delete all documents. Defaults to False.
        """
        if delete_all and self._index is not None:
            self._index.delete(delete_all=True, namespace=self.namespace)
            self._index = self.connect_to_index()
        else:
            if not document_ids:
                logger.warning(
                    "No document IDs provided. No documents will be deleted."
                )
            else:
                self._index.delete(ids=document_ids, namespace=self.namespace)

    def delete_documents_by_filters(self, filters: dict[str, Any], top_k: int = 1000) -> None:
        """
        Delete documents from the Pinecone vector store using filters.

        Args:
            filters (dict[str, Any]): Filters to select documents to delete.
            top_k (int): Maximum number of documents to retrieve for deletion. Defaults to 1000.
        """
        if self.index_type is None or self.index_type == PineconeIndexType.SERVERLESS:
            """
            Serverless and Starter indexes do not support deleting with metadata filtering.
            """
            documents = self._embedding_retrieval(
                query_embedding=self._dummy_vector,
                filters=filters,
                exclude_document_embeddings=True,
                top_k=top_k,
            )
            document_ids = [doc.id for doc in documents]
            self.delete_documents(document_ids=document_ids)
        else:
            filters = _normalize_filters(filters)
            self._index.delete(filter=filters, namespace=self.namespace)

    def delete_documents_by_file_id(self, file_id: str):
        """
        Delete documents from the Pinecone vector store by file ID.
            file_id should be located in the metadata of the document.

        Args:
            file_id (str): The file ID to filter by.
        """
        filters = create_file_id_filter(file_id)
        self.delete_documents_by_filters(filters)

    def list_documents(self, include_embeddings: bool = False, content_key: str | None = None) -> list[Document]:
        """
        List documents in the Pinecone vector store.

        Args:
            include_embeddings (bool): Whether to include embeddings in the results. Defaults to False.
            content_key (Optional[str]): The field used to store content in the storage.

        Returns:
            list[Document]: List of Document objects retrieved.
        """

        all_documents = []
        for batch_doc_ids in self._index.list(namespace=self.namespace):
            response = self._index.fetch(ids=batch_doc_ids, namespace=self.namespace)

            documents = []
            for pinecone_doc in response["vectors"].values():
                content = pinecone_doc["metadata"].pop(content_key or self.content_key, "")

                embedding = None
                if include_embeddings and pinecone_doc["values"] != self._dummy_vector:
                    embedding = pinecone_doc["values"]

                doc = Document(
                    id=pinecone_doc["id"],
                    content=content,
                    metadata=pinecone_doc["metadata"],
                    embedding=embedding,
                    score=None,
                )
                documents.append(doc)

            all_documents.extend(documents)
        return all_documents

    def count_documents(self) -> int:
        """
        Count the number of documents in the store.

        Returns:
            int: The number of documents in the store.
        """
        try:
            count = self._index.describe_index_stats()["namespaces"][self.namespace][
                "vector_count"
            ]
        except KeyError:
            count = 0
        return count

    def write_documents(self, documents: list[Document], content_key: str | None = None) -> int:
        """
        Write documents to the Pinecone vector store.

        Args:
            documents (list[Document]): List of Document objects to write.
            content_key (Optional[str]): The field used to store content in the storage.

        Returns:
            int: Number of documents successfully written.

        Raises:
            ValueError: If documents are not of type Document.
        """
        if len(documents) > 0 and not isinstance(documents[0], Document):
            msg = "param 'documents' must contain a list of objects of type Document"
            raise ValueError(msg)

        documents_for_pinecone = self._convert_documents_to_pinecone_format(
            documents, content_key=content_key or self.content_key
        )

        result = self._index.upsert(
            vectors=documents_for_pinecone,
            namespace=self.namespace,
            batch_size=self.batch_size,
        )

        written_docs = result["upserted_count"]
        return written_docs

    def _convert_documents_to_pinecone_format(
        self,
        documents: list[Document],
        content_key: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Convert Document objects to Pinecone-compatible format.

        Args:
            documents (list[Document]): List of Document objects to convert.
            content_key (Optional[str]): The field used to store content in the storage.

        Returns:
            list[dict[str, Any]]: List of documents in Pinecone-compatible format.
        """
        documents_for_pinecone = []
        for document in documents:
            embedding = copy(document.embedding)
            if embedding is None:
                logger.warning(
                    f"Document {document.id} has no embedding. A dummy embedding will be used."
                )
                embedding = self._dummy_vector
            doc_for_pinecone = {
                "id": document.id,
                "values": embedding,
                "metadata": dict(document.metadata),
            }

            if document.content is not None:
                doc_for_pinecone["metadata"][content_key] = document.content

            documents_for_pinecone.append(doc_for_pinecone)
        return documents_for_pinecone

    def _embedding_retrieval(
        self,
        query_embedding: list[float],
        *,
        namespace: str | None = None,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        exclude_document_embeddings: bool = True,
        content_key: str | None = None,
    ) -> list[Document]:
        """
        Retrieve documents similar to the given query embedding.

        Args:
            query_embedding (list[float]): The query embedding vector.
            namespace (str | None): The namespace to query. Defaults to None.
            filters (dict[str, Any] | None): Filters for the query. Defaults to None.
            top_k (int): Maximum number of documents to retrieve. Defaults to 10.
            exclude_document_embeddings (bool): Whether to exclude embeddings in results. Defaults to True.
            content_key (Optional[str]): The field used to store content in the storage.

        Returns:
            list[Document]: List of retrieved Document objects.

        Raises:
            ValueError: If query_embedding is empty or filter format is incorrect.
        """
        if not query_embedding:
            msg = "query_embedding must be a non-empty list of floats"
            raise ValueError(msg)

        filters = _normalize_filters(filters) if filters else None

        result = self._index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=namespace or self.namespace,
            filter=filters,
            include_values=not exclude_document_embeddings,
            include_metadata=True,
        )

        return self._convert_query_result_to_documents(result, content_key=content_key or self.content_key)

    def _convert_query_result_to_documents(
        self, query_result: dict[str, Any], content_key: str | None = None
    ) -> list[Document]:
        """
        Convert Pinecone query results to Document objects.

        Args:
            query_result (dict[str, Any]): The query result from Pinecone.
            content_key (Optional[str]): The field used to store content in the storage.

        Returns:
            list[Document]: List of Document objects created from the query result.
        """
        pinecone_docs = query_result["matches"]
        documents = []
        for pinecone_doc in pinecone_docs:
            content = pinecone_doc["metadata"].pop(content_key, "")

            embedding = None
            if pinecone_doc["values"] != self._dummy_vector:
                embedding = pinecone_doc["values"]

            doc = Document(
                id=pinecone_doc["id"],
                content=content,
                metadata=pinecone_doc["metadata"],
                embedding=embedding,
                score=pinecone_doc["score"],
            )
            documents.append(doc)

        return documents
