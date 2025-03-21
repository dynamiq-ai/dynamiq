import math
from collections import Counter
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.memory.backends.base import MemoryBackend
from dynamiq.prompts import Message


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
        if not self.documents:
            return 0.0
        total_length = sum(len(doc.lower().split()) for doc in self.documents)
        return total_length / len(self.documents)

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
    def to_dict_exclude_params(self) -> dict[str, bool]:
        """Define parameters to exclude during serialization."""
        return {"messages": True}

    def to_dict(self, include_secure_params: bool = False, **kwargs) -> dict[str, Any]:
        """Converts the instance to a dictionary."""
        return super().to_dict(include_secure_params=include_secure_params, **kwargs)

    def add(self, message: Message) -> None:
        """
        Adds a message to the in-memory list.

        Args:
            message: Message to add to storage

        Raises:
            MemoryBackendError: If the message cannot be added
        """
        self.messages.append(message)

    def get_all(self, limit: int | None = None) -> list[Message]:
        """
        Retrieves all messages from the in-memory list.

        Args:
            limit: Maximum number of messages to return. If provided, returns the most recent messages.
                  If None, returns all messages.

        Returns:
            List of messages sorted by timestamp (oldest first)
        """
        # Sort messages by timestamp
        sorted_messages = sorted(self.messages, key=lambda msg: msg.metadata.get("timestamp", 0))

        # Apply limit if provided
        if limit and len(sorted_messages) > limit:
            return sorted_messages[-limit:]

        return sorted_messages

    def _apply_filters(self, messages: list[Message], filters: dict[str, Any] | None = None) -> list[Message]:
        """
        Applies metadata filters to the list of messages.

        Args:
            messages: List of messages to filter
            filters: Metadata filters to apply

        Returns:
            Filtered list of messages
        """
        if not filters:
            return messages

        filtered_messages = messages
        for key, value in filters.items():
            if isinstance(value, list):
                filtered_messages = [msg for msg in filtered_messages if msg.metadata.get(key) in value]
            else:
                filtered_messages = [msg for msg in filtered_messages if msg.metadata.get(key) == value]

        return filtered_messages

    def search(
        self, query: str | None = None, filters: dict[str, Any] | None = None, limit: int | None = None
    ) -> list[Message]:
        """
        Searches for messages using BM25 scoring, with optional filters.

        Args:
            query: Search query string (optional)
            filters: Optional metadata filters to apply
            limit: Maximum number of messages to return. If None, returns all matching messages.

        Returns:
            List of messages sorted by relevance score (highest first)

        Raises:
            MemoryBackendError: If the search operation fails
        """
        # Apply filters first to reduce search space
        filtered_messages = self._apply_filters(self.messages, filters)

        # If no query provided, return filtered messages
        if not query:
            sorted_messages = sorted(filtered_messages, key=lambda msg: msg.metadata.get("timestamp", 0))
            if limit:
                return sorted_messages[-limit:]
            return sorted_messages

        # Perform BM25 search with query
        query_terms = query.lower().split()
        document_texts = [msg.content for msg in filtered_messages]

        # Handle empty document list
        if not document_texts:
            return []

        # Calculate BM25 scores
        bm25 = BM25DocumentRanker(documents=document_texts)
        scored_messages = [(msg, bm25.score(query_terms, msg.content)) for msg in filtered_messages]

        # Filter out zero scores
        scored_messages = [(msg, score) for msg, score in scored_messages if score > 0]

        # Sort by score (descending)
        scored_messages.sort(key=lambda x: x[1], reverse=True)

        # Apply limit
        result = [msg for msg, _ in scored_messages]
        if limit:
            return result[:limit]

        return result

    def is_empty(self) -> bool:
        """
        Checks if the in-memory list is empty.

        Returns:
            True if the memory is empty, False otherwise
        """
        return len(self.messages) == 0

    def clear(self) -> None:
        """
        Clears the in-memory list.

        Raises:
            MemoryBackendError: If the memory cannot be cleared
        """
        self.messages = []
