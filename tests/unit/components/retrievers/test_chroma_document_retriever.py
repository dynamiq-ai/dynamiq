from unittest.mock import MagicMock

from dynamiq.components.retrievers.chroma import ChromaDocumentRetriever
from dynamiq.storages.vector import ChromaVectorStore


class TestChromaDocumentRetriever:
    def test_run_method(self, mock_documents):
        mock_vector_store = MagicMock(spec=ChromaVectorStore)
        mock_vector_store.search_embeddings.return_value = [mock_documents]

        retriever = ChromaDocumentRetriever(vector_store=mock_vector_store, filters={"field": "value"}, top_k=5)

        result = retriever.run(
            query_embedding=[0.1, 0.2, 0.3],
            exclude_document_embeddings=True,
            top_k=2,
            filters={"new_field": "new_value"},
        )

        mock_vector_store.search_embeddings.assert_called_once_with(
            query_embeddings=[[0.1, 0.2, 0.3]], filters={"new_field": "new_value"}, top_k=2
        )

        assert result == {"documents": mock_documents}

    def test_run_method_with_defaults(self, mock_documents, mock_filters):
        mock_vector_store = MagicMock(spec=ChromaVectorStore)
        mock_vector_store.search_embeddings.return_value = [mock_documents]

        retriever = ChromaDocumentRetriever(vector_store=mock_vector_store, filters=mock_filters, top_k=5)

        result = retriever.run(query_embedding=[0.1, 0.2, 0.3])

        mock_vector_store.search_embeddings.assert_called_once_with(
            query_embeddings=[[0.1, 0.2, 0.3]], filters=mock_filters, top_k=5
        )

        assert result == {"documents": mock_documents}
