from typing import Any


def validate_embedding(embedding: Any) -> tuple[bool, str]:
    """
    Validates that an embedding is not empty.

    Args:
        embedding: The embedding vector to validate

    Returns:
        Tuple containing:
            - Boolean indicating if embedding is valid
            - String containing error message if invalid, empty string if valid
    """
    try:
        if embedding is None:
            return False, "Embedding is None"

        if len(embedding) == 0:
            return False, "Embedding is empty (zero length)"
    except (TypeError, AttributeError):
        return False, "Embedding has no length attribute or is not iterable"

    return True, ""


def validate_document_embeddings(documents: Any) -> tuple[bool, str]:
    """
    Validates embeddings for a list of documents.

    Args:
        documents: List of documents with embeddings

    Returns:
        Tuple containing:
            - Boolean indicating if all document embeddings are valid
            - String containing error message if invalid, empty string if valid
    """
    if not documents:
        return True, ""

    try:
        for i, doc in enumerate(documents):
            if not hasattr(doc, "embedding") or doc.embedding is None:
                return False, f"Document at index {i} has no embedding"

            is_valid, error_msg = validate_embedding(doc.embedding)
            if not is_valid:
                return False, f"Document at index {i}: {error_msg}"
    except (TypeError, AttributeError):
        return False, "Documents is not iterable or has incorrect structure"

    return True, ""
