from dynamiq.types import Document
from dynamiq.utils.logger import logger


class DocumentRanker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        use_fp16=True,
        normalize_score: bool = True,
        threshold: float = 0.0,
        top_k: int = 5,
    ):
        from FlagEmbedding import FlagReranker

        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.normalize_score = normalize_score
        self.threshold = threshold
        self.top_k = top_k
        self.reranker = FlagReranker(model_name, use_fp16)

    def run(self, query: str, documents: list[Document]) -> list[Document]:
        """
        Ranks a list of documents based on the query.

        Args:
            query (str): The query to rank the documents against.
            documents (List[Document]): A list of Document instances to rank.

        Returns:
            List[Document]: A list of Document instances sorted by their relevance to the query.
        """
        document_texts = [document.content for document in documents]
        pairs = [[query, document_text] for document_text in document_texts]
        scores = self.reranker.compute_score(pairs, normalize=self.normalize_score)

        for score, doc in zip(scores, documents):
            doc.score = score

        ranked_documents = sorted(
            zip(documents, scores), key=lambda x: x[1], reverse=True
        )
        logger.debug(f"Ranked {len(ranked_documents)} documents")

        selected_documents = [
            document for document, score in ranked_documents if score > self.threshold
        ][: self.top_k]
        logger.debug(
            f"Selected {len(selected_documents)} documents with score > {self.threshold}"
        )

        return selected_documents
