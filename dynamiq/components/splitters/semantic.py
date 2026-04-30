import enum
import re
from copy import deepcopy
from typing import Callable

import numpy as np

from dynamiq.types import Document
from dynamiq.utils.logger import logger


class BreakpointThresholdType(str, enum.Enum):
    PERCENTILE = "percentile"
    STANDARD_DEVIATION = "standard_deviation"
    INTERQUARTILE = "interquartile"
    GRADIENT = "gradient"


_DEFAULT_THRESHOLD_AMOUNT: dict[BreakpointThresholdType, float] = {
    BreakpointThresholdType.PERCENTILE: 95.0,
    BreakpointThresholdType.STANDARD_DEVIATION: 3.0,
    BreakpointThresholdType.INTERQUARTILE: 1.5,
    BreakpointThresholdType.GRADIENT: 95.0,
}


class SemanticChunkerComponent:
    """Semantic-similarity chunker.

    Sentence-splits the text, embeds groups of neighboring sentences, then breaks
    where consecutive embeddings diverge above a configurable threshold.

    ``embed_fn`` must accept ``list[str]`` and return ``list[list[float]]``.
    """

    def __init__(
        self,
        embed_fn: Callable[[list[str]], list[list[float]]],
        breakpoint_threshold_type: BreakpointThresholdType = BreakpointThresholdType.PERCENTILE,
        breakpoint_threshold_amount: float | None = None,
        number_of_chunks: int | None = None,
        buffer_size: int = 1,
        sentence_split_regex: str = r"(?<=[.?!])\s+",
        min_chunk_size: int = 0,
    ) -> None:
        self.embed_fn = embed_fn
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = (
            breakpoint_threshold_amount
            if breakpoint_threshold_amount is not None
            else _DEFAULT_THRESHOLD_AMOUNT[breakpoint_threshold_type]
        )
        self.number_of_chunks = number_of_chunks
        if buffer_size < 0:
            raise ValueError("buffer_size must be >= 0.")
        self.buffer_size = buffer_size
        self.sentence_split_regex = sentence_split_regex
        self.min_chunk_size = min_chunk_size

    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        if not isinstance(documents, list):
            raise TypeError("SemanticChunker expects a list of Documents as input.")
        results: list[Document] = []
        for doc in documents:
            if doc.content is None:
                raise ValueError(f"SemanticChunker requires text content; document ID {doc.id} has none.")
            chunks = self.split_text(doc.content)
            for index, chunk in enumerate(chunks):
                metadata = deepcopy(doc.metadata) if doc.metadata else {}
                metadata["source_id"] = doc.id
                metadata["chunk_index"] = index
                results.append(Document(content=chunk, metadata=metadata))
        logger.debug(f"SemanticChunker: split {len(documents)} documents into {len(results)} chunks.")
        return {"documents": results}

    def split_text(self, text: str) -> list[str]:
        sentences = [piece for piece in re.split(self.sentence_split_regex, text) if piece.strip()]
        if len(sentences) <= 1:
            return sentences

        groups = self._build_groups(sentences)
        embeddings = self.embed_fn(groups)
        if len(embeddings) != len(groups):
            raise ValueError("embed_fn must return one embedding per input string.")
        matrix = np.asarray(embeddings, dtype=float)
        distances = self._pairwise_cosine_distances(matrix).tolist()
        if not distances:
            return ["".join(sentences)]

        breakpoints = self._compute_breakpoints(distances)
        chunks: list[str] = []
        start = 0
        for breakpoint in breakpoints:
            chunks.append(" ".join(sentences[start : breakpoint + 1]).strip())
            start = breakpoint + 1
        chunks.append(" ".join(sentences[start:]).strip())
        chunks = [chunk for chunk in chunks if chunk]

        if self.min_chunk_size:
            merged: list[str] = []
            for chunk in chunks:
                if merged and len(chunk) < self.min_chunk_size:
                    merged[-1] = f"{merged[-1]} {chunk}".strip()
                else:
                    merged.append(chunk)
            chunks = merged
        return chunks

    def _build_groups(self, sentences: list[str]) -> list[str]:
        groups: list[str] = []
        for index in range(len(sentences)):
            start = max(0, index - self.buffer_size)
            end = min(len(sentences), index + self.buffer_size + 1)
            groups.append(" ".join(sentences[start:end]))
        return groups

    def _compute_breakpoints(self, distances: list[float]) -> list[int]:
        if self.number_of_chunks is not None and self.number_of_chunks > 1:
            count = min(self.number_of_chunks - 1, len(distances))
            indexed = sorted(enumerate(distances), key=lambda item: item[1], reverse=True)
            return sorted(index for index, _ in indexed[:count])

        threshold = self._compute_threshold(distances)
        return [index for index, distance in enumerate(distances) if distance > threshold]

    def _compute_threshold(self, distances: list[float]) -> float:
        values = np.asarray(distances, dtype=float)
        if self.breakpoint_threshold_type == BreakpointThresholdType.PERCENTILE:
            return float(np.percentile(values, self.breakpoint_threshold_amount))
        if self.breakpoint_threshold_type == BreakpointThresholdType.STANDARD_DEVIATION:
            return float(values.mean() + self.breakpoint_threshold_amount * values.std())
        if self.breakpoint_threshold_type == BreakpointThresholdType.INTERQUARTILE:
            q1, q3 = np.percentile(values, [25, 75])
            return float(q3 + self.breakpoint_threshold_amount * (q3 - q1))
        if self.breakpoint_threshold_type == BreakpointThresholdType.GRADIENT:
            gradients = np.gradient(values) if values.size > 1 else np.zeros(1)
            return float(np.percentile(gradients, self.breakpoint_threshold_amount))
        raise ValueError(f"Unknown breakpoint_threshold_type: {self.breakpoint_threshold_type}.")

    @staticmethod
    def _pairwise_cosine_distances(matrix: np.ndarray) -> np.ndarray:
        if matrix.shape[0] < 2:
            return np.empty(0, dtype=float)
        left = matrix[:-1]
        right = matrix[1:]
        norms = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            similarities = np.where(norms > 0, np.einsum("ij,ij->i", left, right) / norms, 0.0)
        return 1.0 - similarities
