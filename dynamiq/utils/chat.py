def format_chat_history(chat_history: list[dict[str, str]]) -> str:
    """Format chat history for the orchestrator.

    Args:
        chat_history (list[dict[str, str]]): List of chat entries.

    Returns:
        str: Formatted chat history.
    """
    formatted_history = ""
    for entry in chat_history:
        role = entry["role"].title()
        content = entry["content"]
        formatted_history += f"{role}: {content}\n"
    return formatted_history
