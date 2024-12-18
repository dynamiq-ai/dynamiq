import math
from collections import Counter

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.memory.backends.base import MemoryBackend
from dynamiq.prompts import Message


class InMemoryError(Exception):
    """Base exception class for InMemory backend errors."""
    pass


class BM25DocumentRanker(BaseModel):
    """BM25 implementation for scoring documents."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    documents: list[str]
    k1: float = 1.5
    b: float = 0.75
    avg_dl: float = 0.0

    def model_post_init(self, __context) -> None:
        """Initialize average document length after model creation."""
        self.avg_dl = self._calculate_avg_dl()

    def _calculate_avg_dl(self) -> float:
        """Calculates the average document length (number of terms per document)."""
        total_length = sum(len(doc.lower().split()) for doc in self.documents)
        return total_length / len(self.documents) if self.documents else 0

    def _idf(self, term: str, N: int, df: int) -> float:
        """Calculates the IDF (inverse document frequency) of a term."""
        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def score(self, query_terms: list[str], document: str) -> float:
        """Calculates the BM25 score for a document."""
        doc_terms = document.lower().split()
        doc_len = len(doc_terms)
        doc_term_freqs = Counter(doc_terms)
        N = len(self.documents)
        score = 0.0

        for term in query_terms:
            term_freq = doc_term_freqs.get(term, 0)
            if term_freq == 0:
                continue
            df = sum(1 for doc in self.documents if term in doc.lower().split())
            idf = self._idf(term, N, df)
            numerator = term_freq * (self.k1 + 1)
            denominator = term_freq + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_dl))
            score += idf * (numerator / denominator)

        return score


class InMemory(MemoryBackend):
    """In-memory implementation of the memory storage backend."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "InMemory"
    messages: list[Message] = Field(default_factory=list)

    @property
    def to_dict_exclude_params(self):
        """Define parameters to exclude during serialization."""
        return {"messages": True}

    def to_dict(self, include_secure_params: bool = False, **kwargs) -> dict:
        """Converts the instance to a dictionary."""
        return super().to_dict(include_secure_params=include_secure_params, **kwargs)

    def add(self, message: Message) -> None:
        """Adds a message to the in-memory list."""
        try:
            self.messages.append(message)
        except Exception as e:
            raise InMemoryError(f"Error adding message to InMemory backend: {e}") from e

    def get_all(self) -> list[Message]:
        """Retrieves all messages from the in-memory list."""
        return sorted(self.messages, key=lambda msg: msg.metadata.get("timestamp", 0))

    def _apply_filters(self, messages: list[Message], filters: dict | None = None) -> list[Message]:
        """Applies metadata filters to the list of messages."""
        if not filters:
            return messages
        filtered_messages = messages
        for key, value in filters.items():
            if isinstance(value, list):
                filtered_messages = [msg for msg in filtered_messages if any(v == msg.metadata.get(key) for v in value)]
            else:
                filtered_messages = [msg for msg in filtered_messages if value == msg.metadata.get(key)]
        return filtered_messages

    def search(self, query: str | None = None, limit: int = 10, filters: dict | None = None) -> list[Message]:
        """Searches for messages using BM25 scoring, with optional filters."""
        if not query and not filters:
            return self.get_all()[:limit]

        filtered_messages = self._apply_filters(self.messages, filters)
        if not query:
            return filtered_messages[:limit]

        query_terms = query.lower().split()
        document_texts = [msg.content for msg in filtered_messages]

        bm25 = BM25DocumentRanker(documents=document_texts)
        scored_messages: list[tuple[Message, float]] = [
            (msg, bm25.score(query_terms, msg.content)) for msg in filtered_messages
        ]
        scored_messages = [(msg, score) for msg, score in scored_messages if score > 0]
        scored_messages.sort(key=lambda x: x[1], reverse=True)

        return [msg for msg, _ in scored_messages][:limit]

    def is_empty(self) -> bool:
        """Checks if the in-memory list is empty."""
        return len(self.messages) == 0

    def clear(self) -> None:
        """Clears the in-memory list."""
        self.messages = []
