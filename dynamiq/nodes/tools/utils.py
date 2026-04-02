import io
import json
import mimetypes
import os
import re
from pathlib import Path

import filetype
import requests


def sanitize_filename(filename: str | None, default: str | None = None) -> str:
    """
    Sanitize a filename to prevent path traversal attacks.

    Removes directory components, path traversal sequences, null bytes,
    and other potentially dangerous characters.

    Args:
        filename: The raw filename to sanitize.
        default: Optional default value to return if filename is empty after sanitization.
            If None, returns empty string for empty input.

    Returns:
        A safe filename string without path components or dangerous characters.
    """
    if not filename:
        return default if default is not None else ""

    filename = os.path.basename(filename)
    filename = filename.replace("\x00", "")
    filename = filename.lstrip(".")
    # Replace path separators and other dangerous characters
    filename = re.sub(r"[/\\]", "_", filename)
    # Remove any remaining control characters
    filename = re.sub(r"[\x00-\x1f\x7f]", "", filename)

    filename = filename.strip()

    if not filename and default is not None:
        return default

    return filename


def guess_mime_type_from_bytes(data: bytes, filename: str = None) -> str:
    """
    Automatically detect MIME type from bytes data using multiple methods.

    Args:
        data: The file content as bytes
        filename: Optional filename to help with MIME type detection

    Returns:
        str: The detected MIME type, defaults to 'application/octet-stream' if unknown
    """
    if filename:
        mime_type, _ = mimetypes.guess_type(filename)
        if mime_type:
            return mime_type

    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    elif data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    elif data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "image/gif"
    elif data.startswith(b"%PDF-"):
        return "application/pdf"
    elif data.startswith(b"PK\x03\x04") or data.startswith(b"\x50\x4b\x03\x04"):
        return "application/zip"
    elif data.startswith(b"<!DOCTYPE html") or data.startswith(b"<html"):
        return "text/html"
    elif data.startswith(b"<?xml"):
        return "application/xml"
    elif data.startswith(b"{") or data.startswith(b"["):
        try:
            json.loads(data.decode("utf-8"))
            return "application/json"
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
    elif data.startswith(b"Name,") or b"," in data[:100] and b"\n" in data[:100]:
        return "text/csv"
    elif data.startswith(b"#") or (b" " in data and b"\n" in data and len(data.split(b"\n")) > 1):
        return "text/plain"

    # Default fallback
    return "application/octet-stream"


def create_file_from_url(
    file_url: str,
    original_name: str | None = None,
    timeout: float | None = None,
) -> io.BytesIO:
    """Create a properly configured BytesIO file from image bytes.

    Args:
        file_url: Signed download URL.
        original_name: Optional original filename to base the new name on.
                       If provided, preserves the original extension when available.
                       If None, will use generic names like "file.png" or "file".
        timeout: Optional timeout for file download.

    Returns:
        BytesIO object with name and content_type attributes.
    """
    file_bytes, content_type_header = download_file_from_url(file_url, timeout)
    file = io.BytesIO(file_bytes)

    ext = ""
    if original_name:
        original_path = Path(original_name)
        if original_path.suffix:
            ext = original_path.suffix
    content_type = content_type_header

    if not ext or not content_type:
        kind = filetype.guess(file_bytes)
        if kind:
            ext = ext or f".{kind.extension}"
            content_type = content_type or kind.mime

    if original_name:
        base_name = Path(original_name).stem
        file.name = f"{base_name}{ext}"
    else:
        file.name = f"file{ext}"

    file.content_type = content_type
    return file


def download_file_from_url(url: str, timeout: float = 60) -> tuple[bytes, str | None]:
    """Download file from URL and return bytes with content type.

    Args:
        url: Presigned url to download.
        timeout: Optional timeout for file download.

    Returns:
        Tuple of (file bytes, content-type header value or None).
    """
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    content_type = response.headers.get("content-type")
    return response.content, content_type
