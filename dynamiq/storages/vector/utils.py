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
