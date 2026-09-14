"""How stored-file inputs are described to the LLM.

A ``map_from_storage`` field is filled by the agent from the file store or the sandbox: the LLM
only ever names a file. The prompt must say that, whatever Python type the field happens to
accept, or the model is invited to type file contents by hand.
"""

import io

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.nodes.agents.components.schema_generator import generate_input_formats


class _Schema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    audio: io.BytesIO | bytes | str | list[io.BytesIO | bytes | str] | None = Field(
        default=None,
        description="Audio file to transcribe.",
        json_schema_extra={"map_from_storage": True},
    )


class _Tool:
    name = "speech-to-text"
    resolved_input_schema = _Schema


def test_a_stored_file_union_is_described_as_a_file_name():
    formats = generate_input_formats([_Tool()], sanitize_tool_name=lambda name: name)

    assert "audio (tuple[str, ...])" in formats
    assert "BytesIO" not in formats
