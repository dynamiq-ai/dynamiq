import os
import re

import requests

from dynamiq.utils.logger import logger

ENDPOINT_REPHRASE = "https://cce4b55a-bd05-46c5-a512-a601b092ff82.apps.sandbox.getdynamiq.ai"  # for example
ENDPOINT_SEARCH = "https://cb7f4ed9-d2e0-4cc2-84be-b56d39daa859.apps.sandbox.getdynamiq.ai"  # for example
ENDPOINT_ANSWER = "https://fc150f29-af76-47f6-8f1c-352cd8cb8660.apps.sandbox.getdynamiq.ai"  # for example
token = os.getenv("DYNAMIQ_API_KEY")

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {token}",
}


def extract_answer(text):
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    return None


def extract_source(text):
    answer_match = re.search(r"<sources>(.*?)</sources>", text, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    return None


def process_query(query: str):
    """
    Generator that processes the query and streams the result chunk by chunk.
    This function yields each chunk to simulate real-time streaming of data.
    """
    # Run the workflow with the provided query
    payload = {
        "input": {
            "input": query,
        }
    }
    response_refactor = requests.post(ENDPOINT_REPHRASE, json=payload, headers=headers, stream=False)  # nosec B113
    response_refactor = response_refactor.json()["output"]
    logger.info(f"response_refactor: {response_refactor}")
    payload_search = {
        "input": {
            "input": response_refactor,
        }
    }

    response_search = requests.post(ENDPOINT_SEARCH, json=payload_search, headers=headers, stream=False)  # nosec B113
    response_search = response_search.json()["output"]["result"]
    logger.info(f"response_search: {response_search}")

    payload_answer = {
        "input": {
            "input": response_search + "\n" + response_refactor,
        }
    }

    response_answer = requests.post(ENDPOINT_ANSWER, json=payload_answer, headers=headers, stream=False)  # nosec B113

    response_answer = response_answer.json()["output"]
    logger.info(f"response_answer: {response_answer}")

    result_answer = extract_answer(response_answer)
    result_sources = extract_source(response_answer)

    result_answer_chunks = result_answer.split(" ")
    result_sources_chunks = [chunk + "\n\n" for chunk in result_sources.split("\n")]

    yield from ["Sources:\n\n"]
    yield from result_sources_chunks
    yield from ["\n\nAnswer:\n\n"]
    yield from result_answer_chunks
