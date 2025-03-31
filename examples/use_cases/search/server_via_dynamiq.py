"""
This script demonstrates how to build an approach using dynamically deployed workflows.
It utilizes workflow from three steps: rephrase, search, and answer.
- The rephrase workflow rephrases the input query using a simple agent.
- The search workflow finds the answer using the SERP scale tool.
- The answer workflow extracts the final answer from the search results and formats it based on the sources.
"""

import os
import re

import requests

from dynamiq.utils.logger import logger

ENDPOINT = os.getenv("DYNAMIQ_ENDPOINT", "Your Dynamiq endpoint")
TOKEN = os.getenv("DYNAMIQ_API_KEY")

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {TOKEN}",
}


def extract_tag_content(text, tag):
    """
    Extract content wrapped within specific XML-like tags from the text.

    Args:
        text (str): The input text containing the tag.
        tag (str): The tag name to extract content from.

    Returns:
        str: The content inside the tag if found, otherwise None.
    """
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None


def process_query(query: str):
    """
    Process the query using the deployed workflows and yield chunks of the result.
    This function simulates real-time streaming of data.

    Args:
        query (str): The query string to process.

    Yields:
        str: Chunks of the final answer and sources.
    """
    try:
        # Step 1: Rephrase the query
        payload = {"input": {"input": query}}
        response = requests.post(ENDPOINT, json=payload, headers=headers)  # nosec B113
        response.raise_for_status()
        final_response = response.json()["output"]
        logger.info(f"Final Response: {final_response}")

        answer = extract_tag_content(final_response, "answer")
        sources = extract_tag_content(final_response, "sources")

        # Stream sources first
        yield "Sources:\n\n"
        for source in sources.split("\n"):
            yield source + "\n\n"

        # Stream the answer
        yield "\n\nAnswer:\n\n"
        for chunk in answer.split(" "):
            yield chunk + " "

    except requests.RequestException as e:
        logger.error(f"HTTP request failed: {e}")
        yield "Error: Unable to process the query."

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        yield "Error: An unexpected error occurred."
