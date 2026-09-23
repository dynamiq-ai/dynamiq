import math
from typing import Any
from uuid import UUID

from dynamiq.utils import REVERSIBLE_MARKERS, decode_reversible, encode_reversible

# The integers orjson writes as JSON numbers; larger ones are kept as text.
_INT64_MIN, _UINT64_MAX = -(2**63), 2**64 - 1


def _encode_dict_key(key: Any) -> str | int | float | bool | None:
    """Ensure dict key is a JSON-compatible primitive."""
    if isinstance(key, UUID):
        return str(key)
    if isinstance(key, (str, int, float, bool, type(None))):
        return key
    return str(key)


def _is_marker(value: Any) -> bool:
    return isinstance(value, dict) and not REVERSIBLE_MARKERS.isdisjoint(value)


def encode_checkpoint_data(obj: Any) -> Any:
    """Recursively pre-encode non-serializable values in a nested structure.

    Operates on raw Python objects (before Pydantic model_dump) so types like
    BytesIO are properly detected and encoded via encode_reversible markers.
    Values JSON cannot hold as they are (integers beyond 64 bits, NaN and infinities,
    tuples, sets, dicts with non-string keys) get markers too, so a resumed run sees exactly
    what the run produced, and so does whatever a model or an object holds. A dict whose own
    keys look like markers is kept as pairs, so decoding does not mistake it for one.
    Dict keys other than primitives are coerced to strings (e.g. UUID).
    """
    if isinstance(obj, int) and not _INT64_MIN <= obj <= _UINT64_MAX:
        return {"__int__": str(obj)}
    if isinstance(obj, float) and not math.isfinite(obj):
        return {"__float__": str(obj)}
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        items = [(_encode_dict_key(k), encode_checkpoint_data(v)) for k, v in obj.items()]
        if all(isinstance(k, str) and k not in REVERSIBLE_MARKERS for k, _ in items):
            return dict(items)
        return {"__dict_items__": [[encode_checkpoint_data(k), v] for k, v in items]}
    if isinstance(obj, list):
        return [encode_checkpoint_data(item) for item in obj]
    if isinstance(obj, tuple):
        return {"__tuple__": [encode_checkpoint_data(item) for item in obj]}
    if isinstance(obj, (set, frozenset)):
        return {"__set__": [encode_checkpoint_data(item) for item in obj]}
    if isinstance(obj, type):
        # A class, such as the type of an error a result holds: kept by name, as results record it.
        return obj.__name__

    encoded = encode_reversible(obj)
    if encoded is obj or _is_marker(encoded):
        return encoded
    # A model's dump, an object's attributes or an enum's value: data like any other.
    return encode_checkpoint_data(encoded)


def decode_checkpoint_data(obj: Any) -> Any:
    """Recursively decode reversible markers back to original Python types, innermost first.

    Needed for deserializers like orjson that don't support json.loads object_hook, and
    decodes in the same order as that hook.
    """
    if isinstance(obj, dict):
        return decode_reversible({k: decode_checkpoint_data(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [decode_checkpoint_data(item) for item in obj]
    return obj
