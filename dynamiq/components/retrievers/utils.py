from __future__ import annotations

from typing import Iterable

from dynamiq.types import Document


def filter_documents_by_threshold(
    documents: Iterable[Document],
    threshold: float | None,
    *,
    higher_is_better: bool,
) -> list[Document]:
    """Filter documents by score threshold while preserving order."""
    if threshold is None:
        return list(documents)

    filtered: list[Document] = []
    for document in documents:
        score = document.score
        if score is None:
            filtered.append(document)
            continue

        if higher_is_better:
            if score >= threshold:
                filtered.append(document)
        elif score <= threshold:
            filtered.append(document)

    return filtered
