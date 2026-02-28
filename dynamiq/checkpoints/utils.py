from typing import Any
from uuid import UUID

from dynamiq.utils import decode_reversible, encode_reversible


def _encode_dict_key(key: Any) -> str | int | float | bool | None:
    """Ensure dict key is a JSON-compatible primitive."""
    if isinstance(key, UUID):
        return str(key)
    if isinstance(key, (str, int, float, bool, type(None))):
        return key
    return str(key)


def encode_checkpoint_data(obj: Any) -> Any:
    """Recursively pre-encode non-serializable values in a nested structure.

    Operates on raw Python objects (before Pydantic model_dump) so types like
    BytesIO are properly detected and encoded via encode_reversible markers.
    Dict keys are coerced to JSON-compatible primitives (e.g. UUID -> str).
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {_encode_dict_key(k): encode_checkpoint_data(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [encode_checkpoint_data(item) for item in obj]
    return encode_reversible(obj)


def decode_checkpoint_data(obj: Any) -> Any:
    """Recursively decode reversible markers back to original Python types.

    Needed for deserializers like orjson that don't support json.loads object_hook.
    """
    if isinstance(obj, dict):
        decoded = decode_reversible(obj)
        if decoded is not obj:
            return decoded
        return {k: decode_checkpoint_data(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decode_checkpoint_data(item) for item in obj]
    return obj
