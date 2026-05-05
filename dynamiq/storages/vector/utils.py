from typing import Any


DEFAULT_SEARCHABLE_TEXT_METADATA_FIELDS: tuple[str, ...] = (
    "file_name",
    "file_path",
    "title",
    "source",
    "url",
)


def normalize_filters(filters: dict[str, Any] | None) -> dict[str, Any] | None:
    """
    Normalize simple metadata filters to Dynamiq's structured filter format.

    Structured filters are passed through unchanged. Simple key-value filters
    such as {"file_type": "ticket"} become an AND group of equality conditions.
    """
    if not filters:
        return None

    if "operator" in filters and "conditions" in filters:
        return filters
    if {"field", "operator", "value"}.issubset(filters):
        return filters

    conditions = []
    for field, value in filters.items():
        operator = "in" if isinstance(value, list) else "=="
        conditions.append({"field": field, "operator": operator, "value": value})

    return {"operator": "AND", "conditions": conditions}


def create_file_id_filter(file_id: str) -> dict:
    """
    Create filters for vector store query based on file_id.

    Args:
        file_id (str): The file ID to filter by.

    Returns:
        dict: The filter conditions.
    """
    return {
        "operator": "AND",
        "conditions": [
            {"field": "file_id", "operator": "==", "value": file_id},
        ],
    }


def create_file_ids_filter(file_ids: list[str]) -> dict:
    """
    Create filters for vector store query based on multiple file_ids.

    Args:
        file_ids (list[str]): The list of file IDs to filter by.

    Returns:
        dict: The filter conditions.
    """
    return {
        "operator": "AND",
        "conditions": [
            {"field": "file_id", "operator": "in", "value": file_ids},
        ],
    }


def create_pgvector_file_id_filter(file_id: str) -> dict:
    """
    Create filters for pgvector query based on file_id.

    Args:
        file_id (str): The file ID to filter by.

    Returns:
        dict: The filter conditions for pgvector.
    """
    return {"field": "metadata.file_id", "operator": "==", "value": file_id}


def create_pgvector_file_ids_filter(file_ids: list[str]) -> dict:
    """
    Create filters for pgvector query based on multiple file_ids.

    Args:
        file_ids (list[str]): The list of file IDs to filter by.

    Returns:
        dict: The filter conditions for pgvector.
    """
    return {"field": "metadata.file_id", "operator": "in", "value": file_ids}
