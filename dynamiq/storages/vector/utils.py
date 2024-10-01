def create_file_id_filter(file_id: str) -> dict:
    """
    Create filters for Pinecone query based on file_id.

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
