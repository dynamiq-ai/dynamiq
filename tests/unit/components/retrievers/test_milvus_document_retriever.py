from unittest.mock import MagicMock

from dynamiq.components.retrievers.milvus import MilvusDocumentRetriever
from dynamiq.storages.vector import MilvusVectorStore
from dynamiq.types import Document


class TestMilvusDocumentRetriever:
    def test_run_method(self):
        mock_documents = [
            Document(id="1", content="Document 1", embedding=[0.1, 0.2, 0.3]),
            Document(id="2", content="Document 2", embedding=[0.4, 0.5, 0.6]),
        ]
        mock_vector_store = MagicMock(spec=MilvusVectorStore)
        mock_vector_store.search_embeddings.return_value = mock_documents

        retriever = MilvusDocumentRetriever(vector_store=mock_vector_store, filters={"field": "value"}, top_k=5)

        result = retriever.run(
            query_embedding=[0.1, 0.2, 0.3],
            exclude_document_embeddings=True,
            top_k=2,
            filters={"new_field": "new_value"},
        )

        mock_vector_store.search_embeddings.assert_called_once_with(
            query_embeddings=[[0.1, 0.2, 0.3]], filters={"new_field": "new_value"}, top_k=2
        )

        expected_documents = [
            Document(id="1", content="Document 1", embedding=None),
            Document(id="2", content="Document 2", embedding=None),
        ]
        assert result == {"documents": expected_documents}

    def test_run_method_with_defaults(self):
        mock_documents = [
            Document(id="1", content="Document 1", embedding=[0.1, 0.2, 0.3]),
            Document(id="2", content="Document 2", embedding=[0.4, 0.5, 0.6]),
        ]
        mock_filters = {"field": "value"}

        mock_vector_store = MagicMock(spec=MilvusVectorStore)
        mock_vector_store.search_embeddings.return_value = mock_documents

        retriever = MilvusDocumentRetriever(vector_store=mock_vector_store, filters=mock_filters, top_k=5)

        result = retriever.run(query_embedding=[0.1, 0.2, 0.3])

        mock_vector_store.search_embeddings.assert_called_once_with(
            query_embeddings=[[0.1, 0.2, 0.3]], filters=mock_filters, top_k=5
        )

        expected_documents = [
            Document(id="1", content="Document 1", embedding=None),
            Document(id="2", content="Document 2", embedding=None),
        ]
        assert result == {"documents": expected_documents}
