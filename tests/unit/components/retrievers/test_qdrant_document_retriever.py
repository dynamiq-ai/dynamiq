from unittest.mock import MagicMock

from dynamiq.components.retrievers.qdrant import QdrantDocumentRetriever
from dynamiq.storages.vector.qdrant import QdrantVectorStore
from dynamiq.types import Document


class TestQdrantDocumentRetriever:

    def test_run_method(self, mock_documents):
        mock_vector_store = MagicMock(spec=QdrantVectorStore)
        mock_vector_store._query_by_embedding.return_value = mock_documents

        retriever = QdrantDocumentRetriever(vector_store=mock_vector_store, filters={"field": "value"}, top_k=5)

        result = retriever.run(
            query_embedding=[0.1, 0.2, 0.3],
            exclude_document_embeddings=True,
            top_k=2,
            filters={"new_field": "new_value"},
        )

        mock_vector_store._query_by_embedding.assert_called_once_with(
            query_embedding=[0.1, 0.2, 0.3],
            filters={"new_field": "new_value"},
            top_k=2,
            return_embedding=False,
            score_threshold=None,
            content_key=None,
        )

        assert result == {"documents": mock_documents}

    def test_run_method_with_defaults(self, mock_documents, mock_filters):
        mock_vector_store = MagicMock(spec=QdrantVectorStore)
        mock_vector_store._query_by_embedding.return_value = mock_documents

        retriever = QdrantDocumentRetriever(vector_store=mock_vector_store, filters=mock_filters, top_k=5)

        result = retriever.run(query_embedding=[0.1, 0.2, 0.3])

        mock_vector_store._query_by_embedding.assert_called_once_with(
            query_embedding=[0.1, 0.2, 0.3],
            filters=mock_filters,
            top_k=5,
            return_embedding=False,
            score_threshold=None,
            content_key=None,
        )

        assert result == {"documents": mock_documents}

    def test_run_applies_similarity_threshold(self):
        high_score_doc = Document(id="1", content="High", score=0.9)
        low_score_doc = Document(id="2", content="Low", score=0.5)

        mock_vector_store = MagicMock(spec=QdrantVectorStore)
        mock_vector_store._query_by_embedding.return_value = [high_score_doc, low_score_doc]

        retriever = QdrantDocumentRetriever(
            vector_store=mock_vector_store,
            filters={"field": "value"},
            top_k=5,
            similarity_threshold=0.8,
        )

        result = retriever.run(query_embedding=[0.1, 0.2, 0.3])

        mock_vector_store._query_by_embedding.assert_called_once_with(
            query_embedding=[0.1, 0.2, 0.3],
            filters={"field": "value"},
            top_k=5,
            return_embedding=False,
            score_threshold=0.8,
            content_key=None,
        )

        assert result == {"documents": [high_score_doc]}
