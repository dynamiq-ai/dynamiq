from copy import deepcopy
from io import BytesIO
from typing import Any

import filetype

from dynamiq.utils.utils import generate_uuid


def build_source_metadata(metadata: dict[str, Any] | None, file_path: str) -> dict[str, Any]:
    """Return independent metadata with a stable file path.

    ``source`` is reserved for the upstream source identifier (typically a UUID),
    so this helper preserves it only when supplied with a non-empty value by the
    caller. Public URLs stay in their original metadata fields, such as
    ``dynamiq_item_source_provider_url`` or ``source_url``.
    """
    result = deepcopy(metadata or {})
    effective_file_path = result.get("file_path") or file_path
    result["file_path"] = str(effective_file_path)
    if not result.get("source"):
        result.pop("source", None)
    return result


def normalize_table_headers(headers: list[str], width: int) -> list[str]:
    """Return stable, non-empty, globally unique table column labels."""
    base_labels = [
        (headers[index].strip() if index < len(headers) else "") or f"column_{index + 1}" for index in range(width)
    ]
    reserved = set(base_labels)
    used: set[str] = set()
    next_suffix: dict[str, int] = {}
    normalized: list[str] = []

    for base_label in base_labels:
        label = base_label
        if label in used:
            suffix = next_suffix.get(base_label, 2)
            label = f"{base_label}_{suffix}"
            while label in used or label in reserved:
                suffix += 1
                label = f"{base_label}_{suffix}"
            next_suffix[base_label] = suffix + 1
        used.add(label)
        normalized.append(label)

    return normalized


def get_filename_for_bytesio(file: BytesIO) -> str:
    """
    Get a filepath for a BytesIO object.

    Args:
        file (BytesIO): The BytesIO object.

    Returns:
        str: A filename for the BytesIO object.

    Raises:
        ValueError: If the file extension couldn't be guessed.
    """
    filename = getattr(file, "name", None)
    if filename is None:
        file_extension = filetype.guess_extension(file)
        if file_extension:
            filename = f"{generate_uuid()}.{file_extension}"
        else:
            raise ValueError(
                "Unable to determine file extension. BytesIO object lacks name and "
                "extension couldn't be guessed."
            )
    return filename
