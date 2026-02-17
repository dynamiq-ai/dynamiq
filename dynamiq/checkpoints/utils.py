from typing import Any

from dynamiq.utils import decode_reversible, encode_reversible


def encode_checkpoint_data(obj: Any) -> Any:
    """Recursively pre-encode non-serializable values in a nested structure.

    Operates on raw Python objects (before Pydantic model_dump) so types like
    BytesIO are properly detected and encoded via encode_reversible markers.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: encode_checkpoint_data(v) for k, v in obj.items()}
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
