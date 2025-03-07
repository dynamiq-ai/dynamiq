import base64
from datetime import date, datetime
from enum import Enum
from io import BytesIO
from json import JSONEncoder, loads
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, PydanticUserError, RootModel

TRUNCATE_LIMIT = 20


def generate_uuid() -> str:
    """
    Generate a UUID4 string.

    Returns:
        str: A string representation of a UUID4.
    """
    return str(uuid4())


def serialize(obj: Any) -> dict[str, Any]:
    """
    Serialize an object to a JSON-compatible dictionary.

    Args:
        obj (Any): The object to be serialized.

    Returns:
        dict[str, Any]: A dictionary representation of the object, suitable for JSON serialization.
    """
    import jsonpickle

    return loads(jsonpickle.encode(obj, unpicklable=False))


def merge(a: Any, b: Any) -> dict[str, Any]:
    """
    Merge two dictionaries or objects.

    Args:
        a (Any): The first dictionary or object.
        b (Any): The second dictionary or object.

    Returns:
        dict[str, Any]: A new dictionary containing the merged key-value pairs from both inputs.
    """
    return {**a, **b}


def encode_bytes(value: bytes) -> str:
    """
    Encode a bytes object to an encoded string.

    Args:
        value (bytes): The bytes object to be encoded.

    Returns:
        str: encoded string representation of the bytes object.
    """
    try:
        return value.decode()
    except UnicodeDecodeError:
        return base64.b64encode(value).decode()


class JsonWorkflowEncoder(JSONEncoder):
    """
    A custom JSON encoder for handling specific object types in workflow serialization.

    This encoder extends the default JSONEncoder to provide custom serialization for Enum, UUID,
    and datetime objects.
    """

    def default(self, obj: Any) -> Any:
        """
        Encode the given object into a JSON-serializable format.

        Args:
            obj (Any): The object to be encoded.

        Returns:
            Any: A JSON-serializable representation of the object.

        Raises:
            TypeError: If the object type is not handled by this encoder or the default encoder.
        """
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, (BytesIO, bytes, Exception)) or callable(obj):
            return format_value(obj)[0]
        return JSONEncoder.default(self, obj)


def format_value(
    value: Any,
    skip_format_types: set = None,
    force_format_types: set = None,
    truncate_enabled: bool = False,
    truncate_limit: int = TRUNCATE_LIMIT,
    **kwargs,
) -> tuple[Any, dict]:
    """Format a value for serialization.

    Args:
        value (Any): The value to format.
        skip_format_types (set, optional): Types to skip formatting.
        force_format_types (set, optional): Types to force formatting.
        truncate_enabled (bool, optional): Whether to apply truncation.
        truncate_limit (int): The maximum allowed length for the value; if exceeded, the value will be truncated.
        **kwargs: Additional keyword arguments.

    Returns:
        Any: Formatted value.
    """
    from dynamiq.nodes.tools.python import PythonInputSchema
    from dynamiq.runnables import RunnableResult

    if skip_format_types is None:
        skip_format_types = set()
    if force_format_types is None:
        force_format_types = set()

    truncate_metadata = {}
    if not isinstance(value, tuple(force_format_types)) and isinstance(
        value, tuple(skip_format_types)
    ):
        return value, truncate_metadata

    if isinstance(value, BytesIO):
        return getattr(value, "name", None) or encode_bytes(value.getvalue()), truncate_metadata
    if isinstance(value, bytes):
        return encode_bytes(value), truncate_metadata

    path = kwargs.get("path", "")
    if isinstance(value, dict):
        formatted_dict = {}
        for k, v in value.items():
            new_path = f"{path}.{k}" if path else k
            formatted_v, sub_metadata = format_value(
                v, skip_format_types, force_format_types, truncate_enabled, path=new_path
            )
            formatted_dict[k] = formatted_v
            truncate_metadata.update(sub_metadata)
        return formatted_dict, truncate_metadata

    if truncate_enabled and isinstance(value, list) and all(isinstance(v, float) for v in value):
        original_length = len(value)
        if original_length > truncate_limit:
            truncated_value = value[:truncate_limit]
            truncate_metadata[path] = {
                "original_length": original_length,
                "truncated_length": len(truncated_value),
            }
            return truncated_value, truncate_metadata

    if isinstance(value, (list, tuple, set)):
        formatted_list = []
        for i, v in enumerate(value):
            new_path = f"{path}[{i}]"
            formatted_v, sub_metadata = format_value(
                v, skip_format_types, force_format_types, truncate_enabled, path=new_path
            )
            formatted_list.append(formatted_v)
            truncate_metadata.update(sub_metadata)

        return type(value)(formatted_list), truncate_metadata

    if isinstance(value, (RunnableResult, PythonInputSchema)):
        return (
            value.to_dict(skip_format_types=skip_format_types, force_format_types=force_format_types),
            truncate_metadata,
        )
    if isinstance(value, BaseModel):
        base_dict = value.to_dict() if hasattr(value, "to_dict") else value.model_dump()

        if truncate_enabled:
            for attr_name, attr_value in base_dict.items():
                if isinstance(attr_value, list) and all(isinstance(x, float) for x in attr_value):
                    metadata_key = f"{path}.{attr_name}" if path else attr_name

                    original_length = len(attr_value)
                    if original_length > truncate_limit:
                        truncated_value = attr_value[:truncate_limit]

                        truncate_metadata[metadata_key] = {
                            "original_length": original_length,
                            "truncated_length": len(truncated_value),
                        }

                        base_dict[attr_name] = truncated_value

        return base_dict, truncate_metadata
    if isinstance(value, Exception):
        recoverable = bool(kwargs.get("recoverable"))
        return {
            "content": f"{str(value)}",
            "error_type": type(value).__name__,
            "recoverable": recoverable,
        }, truncate_metadata
    if callable(value):
        return f"func: {getattr(value, '__name__', str(value))}", truncate_metadata

    try:
        return RootModel[type(value)](value).model_dump(), truncate_metadata
    except PydanticUserError:
        return str(value), truncate_metadata


def deep_merge(source: dict, destination: dict) -> dict:
    """
    Recursively merge dictionaries with proper override behavior.

    Args:
        source: Source dictionary with higher priority values
        destination: Destination dictionary with lower priority values

    Returns:
        dict: Merged dictionary where source values override destination values,
              and lists are concatenated when both source and destination have lists
    """
    result = destination.copy()
    for key, value in source.items():
        if key in result:
            if isinstance(value, dict) and isinstance(result[key], dict):
                result[key] = deep_merge(value, result[key])
            elif isinstance(value, list) and isinstance(result[key], list):
                result[key] = result[key] + value
            else:
                result[key] = value
        else:
            result[key] = value
    return result
