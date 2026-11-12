"""Checks shared by the semantic mapping tests."""

import json
from pathlib import Path

SCHEMAS = Path(__file__).parent / "schemas"


def assert_otel_valid(attributes: dict) -> None:
    """Every value is one an OpenTelemetry attribute accepts: a primitive or a homogeneous list of one."""
    for key, value in attributes.items():
        assert isinstance(key, str) and key, key
        assert value is not None, key
        if isinstance(value, list):
            kinds = {type(item) for item in value}
            assert len(kinds) <= 1, f"{key}: mixed list {value!r}"
            assert kinds <= {str, bool, int, float}, f"{key}: {value!r}"
        else:
            assert type(value) in (str, bool, int, float), f"{key}: {type(value).__name__}"


def _schema(name: str) -> dict:
    return json.loads((SCHEMAS / name).read_text())


def _part_requirements(schema: dict) -> dict[str, list[str]]:
    """Required keys of every part type with a fixed `type` in a message schema."""
    out = {}
    for definition in schema["$defs"].values():
        type_property = definition.get("properties", {}).get("type", {})
        if "const" in type_property and "parts" not in definition.get("properties", {}):
            out[type_property["const"]] = definition.get("required", [])
    return out


def validate_messages(value: str, kind: str) -> list[dict]:
    """Check a message attribute against the v1.41.1 input or output message schema, by hand.

    jsonschema is not a dependency; the checks cover what the schema requires: the message keys, the part keys of
    every typed part and the string fields.
    """
    schema = _schema(f"gen-ai-{kind}-messages.json")
    message_def = schema["$defs"]["ChatMessage" if kind == "input" else "OutputMessage"]
    part_requirements = _part_requirements(schema)
    messages = json.loads(value)
    assert isinstance(messages, list)
    for message in messages:
        assert isinstance(message, dict)
        for key in message_def["required"]:
            assert key in message, f"message lacks {key}: {message}"
        assert isinstance(message["role"], str)
        if "name" in message:
            assert isinstance(message["name"], str)
        if kind == "output":
            assert isinstance(message["finish_reason"], str)
        assert isinstance(message["parts"], list)
        for part in message["parts"]:
            assert isinstance(part, dict) and isinstance(part.get("type"), str), part
            for key in part_requirements.get(part["type"], []):
                assert key in part, f"{part['type']} part lacks {key}: {part}"
            if part["type"] in ("text", "reasoning"):
                assert isinstance(part["content"], str)
            if part["type"] in ("blob", "uri", "file"):
                assert isinstance(part["modality"], str)
            if part["type"] == "blob":
                assert isinstance(part["content"], str)
    return messages
