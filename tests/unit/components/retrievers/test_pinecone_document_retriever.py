from unittest.mock import MagicMock

from dynamiq.components.retrievers.pinecone import PineconeDocumentRetriever
from dynamiq.storages.vector import PineconeVectorStore


class TestPineconeDocumentRetriever:

    def test_run_method(self, mock_documents):
        mock_vector_store = MagicMock(spec=PineconeVectorStore)
        mock_vector_store._embedding_retrieval.return_value = mock_documents

        retriever = PineconeDocumentRetriever(vector_store=mock_vector_store, filters={"field": "value"}, top_k=5)

        result = retriever.run(
            query_embedding=[0.1, 0.2, 0.3],
            exclude_document_embeddings=True,
            top_k=2,
            filters={"new_field": "new_value"},
        )

        mock_vector_store._embedding_retrieval.assert_called_once_with(
            query_embedding=[0.1, 0.2, 0.3],
            filters={"new_field": "new_value"},
            top_k=2,
            exclude_document_embeddings=True,
            content_key=None,
        )

        assert result == {"documents": mock_documents}

    def test_run_method_with_defaults(self, mock_documents, mock_filters):
        mock_vector_store = MagicMock(spec=PineconeVectorStore)
        mock_vector_store._embedding_retrieval.return_value = mock_documents

        retriever = PineconeDocumentRetriever(vector_store=mock_vector_store, filters=mock_filters, top_k=5)

        result = retriever.run(query_embedding=[0.1, 0.2, 0.3])

        mock_vector_store._embedding_retrieval.assert_called_once_with(
            query_embedding=[0.1, 0.2, 0.3],
            filters=mock_filters,
            top_k=5,
            exclude_document_embeddings=True,
            content_key=None,
        )

        assert result == {"documents": mock_documents}
