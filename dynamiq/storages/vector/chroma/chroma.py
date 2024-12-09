from collections import defaultdict
from typing import TYPE_CHECKING, Any, Optional

from dynamiq.connections import Chroma
from dynamiq.storages.vector.utils import create_file_id_filter
from dynamiq.types import Document
from dynamiq.utils.logger import logger

if TYPE_CHECKING:
    from chromadb.api import ClientAPI
    from chromadb.api.types import QueryResult


CHROMA_OPERATOR_MAPPING = {
    "==": "$eq",
    "!=": "$ne",
    ">": "$gt",
    ">=": "$gte",
    "<": "$lt",
    "<=": "$lte",
    "in": "$in",
    "not in": "$nin",
}


class ChromaVectorStore:
    """
    Vector store using Chroma.

    This class provides an interface to interact with a Chroma vector store for document storage and
    retrieval.

    Attributes:
        client (ClientAPI): The Chroma client API instance.
        index_name (str): The name of the index or collection in the vector store.
        _collection: The Chroma collection object.
    """

    def __init__(
        self,
        connection: Chroma | None = None,
        client: Optional["ClientAPI"] = None,
        index_name: str = "default",
        create_if_not_exist: bool = False,
    ):
        """
        Initialize the ChromaVectorStore.

        Args:
            connection (Chroma | None): A Chroma connection object. Defaults to None.
            client (Optional[ClientAPI]): A Chroma client API instance. Defaults to None.
            index_name (str): The name of the index or collection. Defaults to "default".
        """
        self.client = client
        if self.client is None:
            connection = connection or Chroma()
            self.client = connection.connect()
        self.index_name = index_name
        if create_if_not_exist:
            self._collection = self.client.get_or_create_collection(name=index_name)
        else:
            self._collection = self.client.get_collection(name=index_name)

    def count_documents(self) -> int:
        """
        Get the number of documents in the collection.

        Returns:
            int: The number of documents in the collection.
        """
        return self._collection.count()

    def write_documents(self, documents: list[Document]) -> int:
        """
        Write (or overwrite) documents into the store.

        This method processes a list of documents and writes them into the vector store.

        Args:
            documents (list[Document]): A list of Document objects to be written into the document
                store.

        Raises:
            ValueError: If an item in the documents list is not an instance of the Document class.

        Returns:
            int: The number of documents successfully written to the document store.
        """
        for doc in documents:
            if not isinstance(doc, Document):
                msg = (
                    "param 'documents' must contain a list of objects of type Document"
                )
                raise ValueError(msg)

            data = {"ids": [doc.id], "documents": [doc.content]}

            if doc.metadata:
                data["metadatas"] = [doc.metadata]

            if doc.embedding:
                data["embeddings"] = [doc.embedding]

            self._collection.add(**data)

        return len(documents)

    def delete_documents(self, document_ids: list[str] | None = None, delete_all: bool = False) -> None:
        """
        Delete documents from the vector store based on their IDs.

        Args:
            document_ids (list[str]): A list containing the IDs of documents to be deleted from the store.
            delete_all (bool): A flag to delete all documents from the store. Defaults to False.
        """

        if delete_all and self._collection is not None:
            self.client.delete_collection(name=self.index_name)
            self._collection = self.client.get_or_create_collection(
                name=self.index_name
            )
        else:
            if not document_ids:
                logger.warning(
                    "No document IDs provided. No documents will be deleted."
                )
            else:
                self._collection.delete(ids=document_ids)

    def delete_documents_by_filters(self, filters: dict[str, Any] | None = None) -> None:
        """
        Delete documents from the vector store based on the provided filters.

        Args:
            filters (dict[str, Any] | None): The filters to apply to the document list. Defaults to
                None.
        """
        if filters is None:
            raise ValueError("No filters provided. No documents will be deleted.")
        else:
            ids, where, where_document = self._normalize_filters(filters)
            self._collection.delete(ids=ids, where=where, where_document=where_document)

    def delete_documents_by_file_id(self, file_id: str) -> None:
        """
        Delete documents from the vector store based on the provided file ID.
            file_id should be located in the metadata of the document.

        Args:
            file_id (str): The file ID to filter by.
        """
        filters = create_file_id_filter(file_id)
        self.delete_documents_by_filters(filters)

    def search_embeddings(
        self,
        query_embeddings: list[list[float]],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[list[Document]]:
        """
        Perform vector search on the stored documents using query embeddings.

        Args:
            query_embeddings (list[list[float]]): A list of embeddings to use as queries.
            top_k (int): The maximum number of documents to retrieve.
            filters (dict[str, Any] | None): A dictionary of filters to apply to the search.
                Defaults to None.

        Returns:
            list[list[Document]]: A list of lists containing documents that match the given filters,
                for each query embedding provided.
        """
        if filters is None:
            results = self._collection.query(
                query_embeddings=query_embeddings,
                n_results=top_k,
                include=["embeddings", "documents", "metadatas", "distances"],
            )
        else:
            chroma_filters = self._normalize_filters(filters=filters)
            results = self._collection.query(
                query_embeddings=query_embeddings,
                n_results=top_k,
                where=chroma_filters[1],
                where_document=chroma_filters[2],
                include=["embeddings", "documents", "metadatas", "distances"],
            )

        return self._query_result_to_documents(results)

    def filter_documents(self, filters: dict[str, Any] | None = None) -> list[Document]:
        """
        Retrieve documents that match the provided filters.

        Filters can be defined in two formats:
        1. Old format: Nested dictionaries with logical operators and comparison operators.
        2. New format: Nested dictionaries of Comparison and Logic types.

        For the new format:
        Comparison dictionaries must contain the following keys:
        - 'field': The name of one of the metadata fields of a document (e.g., 'metadata.years').
        - 'operator': One of '==', '!=', '>', '>=', '<', '<=', 'in', 'not in'.
        - 'value': A single value or (for 'in' and 'not in') a list of values.

        Logic dictionaries must contain the following keys:
        - 'operator': One of 'NOT', 'OR', 'AND'.
        - 'conditions': A list of Comparison or Logic dictionaries.

        Example of new format:
        {
            "operator": "AND",
            "conditions": [
              {
                "field": "metadata.years",
                "operator": "==",
                "value": "2019"
              },
              {
                "field": "metadata.companies",
                "operator": "in",
                "value": ["BMW", "Mercedes"]
              }
            ]
        }

        Args:
            filters (Dict[str, Any] | None): The filters to apply to the document list.
            filters (dict[str, Any] | None): The filters to apply to the document list. Defaults to
                None.

        Returns:
            list[Document]: A list of Document instances that match the given filters.
        """
        if filters:
            ids, where, where_document = self._normalize_filters(filters)
            kwargs: dict[str, Any] = {"where": where}

            if ids:
                kwargs["ids"] = ids
            if where_document:
                kwargs["where_document"] = where_document

            result = self._collection.get(**kwargs)
        else:
            raise ValueError(
                "No filters provided. No documents will be retrieved with filters."
            )

        return self._get_result_to_documents(result)

    def list_documents(self) -> list[Document]:
        """
        List all documents in the collection.

        Returns:
            list[Document]: A list of Document instances representing all documents in the collection.
        """
        result = self._collection.get()
        return self._get_result_to_documents(result)

    @staticmethod
    def _normalize_filters(
        filters: dict[str, Any]
    ) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
        """
        Translate filters to Chroma filters.

        Args:
            filters (Dict[str, Any]): The filters to normalize.

        Returns:
            Tuple[List[str], Dict[str, Any], Dict[str, Any]]: A tuple containing:
                - A list of document IDs
                - A dictionary of 'where' conditions
                - A dictionary of 'where_document' conditions

        Raises:
            TypeError: If the 'filters' parameter is not a dictionary.
            ValueError: If the filter structure is invalid or contains unsupported operators.
        """
        if not isinstance(filters, dict):
            raise TypeError("'filters' parameter must be a dictionary")

        # Check if it's the new format
        if "operator" in filters or "conditions" in filters:
            processed_filters = ChromaVectorStore._process_filter_node(filters)
        else:
            # It's the old format, use the old processing method
            return ChromaVectorStore._process_old_filters(filters)

        ids = []
        where_document = {}

        # Extract 'id' and 'content' filters if present
        if "metadata.id" in processed_filters:
            ids = processed_filters["metadata.id"].get("$eq", [])
            del processed_filters["metadata.id"]

        if "content" in processed_filters:
            where_document["$contains"] = processed_filters["content"].get("$eq", "")
            del processed_filters["content"]

        where = processed_filters

        if "$and" in where and "$or" not in where:
            and_conditions = where["$and"]
            if len(and_conditions) == 1:
                where = and_conditions[0]
        if "$or" in where and "$and" not in where:
            or_conditions = where["$or"]
            if len(or_conditions) == 1:
                where = or_conditions[0]

        try:
            if where:
                from chromadb.api.types import validate_where

                validate_where(where)
            if where_document:
                from chromadb.api.types import validate_where_document

                validate_where_document(where_document)
        except ValueError as e:
            raise ValueError(e) from e

        return ids, where, where_document

    @staticmethod
    def _process_old_filters(
        filters: dict[str, Any]
    ) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
        """
        Process filters in the old format.
        """
        ids = []
        where = defaultdict(list)
        where_document = defaultdict(list)
        keys_to_remove = []

        for field, value in filters.items():
            if field == "content":
                keys_to_remove.append(field)
                where_document["$contains"] = value
            elif field == "id":
                keys_to_remove.append(field)
                ids.append(value)
            elif isinstance(value, (list, tuple)):
                keys_to_remove.append(field)
                if len(value) == 0:
                    continue
                if len(value) == 1:
                    where[field] = value[0]
                    continue
                for v in value:
                    where["$or"].append({field: v})

        for k in keys_to_remove:
            del filters[k]

        final_where = dict(filters)
        final_where.update(dict(where))

        return ids, final_where, dict(where_document)

    @staticmethod
    def _process_filter_node(node: dict[str, Any]) -> dict[str, Any]:
        """
        Process a single node in the filter structure.

        Args:
            node (Dict[str, Any]): A dictionary representing a filter node.

        Returns:
            Dict[str, Any]: The processed filter node.

        Raises:
            ValueError: If the node structure is invalid.
        """
        if "operator" in node and "conditions" in node:  # Logic node
            return ChromaVectorStore._process_logic_node(node)
        elif (
            "field" in node and "operator" in node and "value" in node
        ):  # Comparison node
            return ChromaVectorStore._process_comparison_node(node)
        else:
            raise ValueError("Invalid filter node structure")

    @staticmethod
    def _process_logic_node(node: dict[str, Any]) -> dict[str, Any]:
        """
        Process a logic node in the filter structure.

        Args:
            node (Dict[str, Any]): A dictionary representing a logic node.

        Returns:
            Dict[str, Any]: The processed logic node.
        """
        operator = node["operator"].lower()
        conditions = [
            ChromaVectorStore._process_filter_node(condition)
            for condition in node["conditions"]
        ]
        return {f"${operator}": conditions}

    @staticmethod
    def _process_comparison_node(node: dict[str, Any]) -> dict[str, Any]:
        """
        Process a comparison node in the filter structure.

        Args:
            node (Dict[str, Any]): A dictionary representing a comparison node.

        Returns:
            Dict[str, Any]: The processed comparison node.

        Raises:
            ValueError: If the operator is unsupported.
        """
        field = node["field"]
        operator = node["operator"]
        value = node["value"]

        chroma_operator = CHROMA_OPERATOR_MAPPING.get(operator)

        if chroma_operator is None:
            raise ValueError(f"Unsupported operator: {operator}")

        return {field: {chroma_operator: value}}

    @staticmethod
    def _query_result_to_documents(result: "QueryResult") -> list[list[Document]]:
        """
        Convert Chroma query results into Dynamiq Documents.

        Args:
            result (QueryResult): The result from a Chroma query operation.

        Returns:
            list[list[Document]]: A list of lists containing Document objects created from the
                Chroma query result.
        """
        return_value: list[list[Document]] = []
        documents = result.get("documents")
        if documents is None:
            return return_value

        for i, answers in enumerate(documents):
            converted_answers = []
            for j in range(len(answers)):
                document_dict: dict[str, Any] = {
                    "id": result["ids"][i][j],
                    "content": documents[i][j],
                }

                if metadatas := result.get("metadatas"):
                    document_dict["metadata"] = dict(metadatas[i][j])

                if embeddings := result.get("embeddings"):
                    document_dict["embedding"] = embeddings[i][j]

                if distances := result.get("distances"):
                    document_dict["score"] = distances[i][j]

                converted_answers.append(Document(**document_dict))
            return_value.append(converted_answers)

        return return_value

    @staticmethod
    def _get_result_to_documents(result: "QueryResult") -> list[Document]:
        """
        Convert Chroma get result into Dynamiq Documents.

        Args:
            result (GetResult): The result from a Chroma get operation.

        Returns:
            list[Document]: A list containing Document objects created from the
                Chroma get result.
        """
        return_value: list[Document] = []
        documents = result.get("documents")
        if documents is None:
            return return_value

        for i, content in enumerate(documents):
            document_dict: dict[str, Any] = {
                "id": result["ids"][i],
                "content": content,
            }

            if metadatas := result.get("metadatas"):
                document_dict["metadata"] = dict(metadatas[i])

            if embeddings := result.get("embeddings"):
                document_dict["embedding"] = embeddings[i]

            if distances := result.get("distances"):
                document_dict["score"] = distances[i]

            return_value.append(Document(**document_dict))
        return return_value
