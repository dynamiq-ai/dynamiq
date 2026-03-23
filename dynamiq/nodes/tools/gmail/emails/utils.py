import base64
from enum import Enum
from typing import Any

from bs4 import BeautifulSoup


class GmailLabel(str, Enum):
    """Enum for supported Gmail system labels."""

    INBOX = "INBOX"
    SPAM = "SPAM"
    TRASH = "TRASH"
    UNREAD = "UNREAD"
    STARRED = "STARRED"
    IMPORTANT = "IMPORTANT"
    CATEGORY_PERSONAL = "CATEGORY_PERSONAL"
    CATEGORY_SOCIAL = "CATEGORY_SOCIAL"
    CATEGORY_PROMOTIONS = "CATEGORY_PROMOTIONS"
    CATEGORY_UPDATES = "CATEGORY_UPDATES"
    CATEGORY_FORUMS = "CATEGORY_FORUMS"


def extract_text_from_html(html_body: str) -> str:
    """
    Extracts plain text from an HTML string using BeautifulSoup.

    Args:
        html_body (str): The HTML content as a string.

    Returns:
        str: The extracted plain text.
    """
    soup = BeautifulSoup(html_body, "html.parser")
    return soup.get_text(separator="\n", strip=True)


def decode_body_data(part: dict[str, Any]) -> str:
    """
    Decodes base64 encoded email body data, extracting text from HTML if needed.

    Args:
        part (dict[str, Any]): A dictionary containing 'body' with base64 'data' and 'mimeType'.

    Returns:
        str: Decoded plain text of the email body.
    """
    raw_bytes = base64.urlsafe_b64decode(part["body"]["data"].encode("UTF-8"))
    raw_text = raw_bytes.decode("utf-8", errors="ignore")
    if part.get("mimeType") == "text/html":
        return extract_text_from_html(raw_text)
    return raw_text


def extract_headers(msg_data: dict[str, Any]) -> dict[str, str]:
    """
    Extracts email headers into a dictionary for easy access.

    Args:
        msg_data (dict[str, Any]): Full Gmail message data from API.

    Returns:
        dict[str, str]: Dictionary of header name to header value.
    """
    return {h["name"]: h["value"] for h in msg_data["payload"].get("headers", [])}


def extract_body(msg_data: dict[str, Any]) -> str:
    """
    Extracts the email body text from the Gmail message payload.

    Args:
        msg_data (dict[str, Any]): Full Gmail message data from API.

    Returns:
        str: Plain text body extracted from message parts or payload.
    """
    parts = msg_data["payload"].get("parts", [])
    if parts:
        for part in parts:
            if "data" in part.get("body", {}):
                if part.get("mimeType") in ["text/plain", "text/html"]:
                    return decode_body_data(part)
    body = msg_data["payload"].get("body", {})
    if "data" in body:
        return decode_body_data({"body": body, "mimeType": msg_data["payload"].get("mimeType", "")})
    return ""
