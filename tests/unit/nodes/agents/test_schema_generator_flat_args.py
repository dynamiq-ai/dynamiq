"""Tests for the flat-args function-calling schema shape.

Verifies that `generate_function_calling_schemas` produces schemas where:
  * `thought` is the FIRST property (load-bearing for streaming UX and model
    chain-of-thought behavior).
  * Tool params are TOP-LEVEL siblings of `thought` (no `action_input` wrapper).
  * `additionalProperties: false` is set on `parameters`.
  * Strict mode is enabled only when every property is required AND the schema
    contains no shapes OpenAI strict mode would reject.
"""

from typing import Optional
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

from dynamiq.nodes.agents.components.schema_generator import generate_function_calling_schemas


def _sanitize(name: str) -> str:
    return name


def _make_tool(name: str, input_schema_cls: type[BaseModel] | None = None, description: str = "desc"):
    """Build a mock tool with a Pydantic input schema."""

    class _NoFields(BaseModel):
        pass

    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.input_schema = input_schema_cls if input_schema_cls is not None else _NoFields
    return tool


class TestFlatArgsSchema:
    def test_thought_is_first_property_for_required_only_tool(self):
        class _Schema(BaseModel):
            file_path: str = Field(..., description="Path")
            content: str = Field(..., description="Content")

        tool = _make_tool("file_write", _Schema)
        schemas = generate_function_calling_schemas(tools=[tool], delegation_allowed=False, sanitize_tool_name=_sanitize)
        tool_schema = next(s for s in schemas if s["function"]["name"] == "file_write")

        properties = tool_schema["function"]["parameters"]["properties"]
        required = tool_schema["function"]["parameters"]["required"]

        assert list(properties.keys()) == ["thought", "file_path", "content"]
        assert required[0] == "thought"
        assert set(required) == {"thought", "file_path", "content"}

    def test_thought_is_first_property_for_mixed_params_tool(self):
        class _Schema(BaseModel):
            query: str = Field(..., description="Query")
            limit: int = Field(default=10, description="Max results")

        tool = _make_tool("search", _Schema)
        schemas = generate_function_calling_schemas(tools=[tool], delegation_allowed=False, sanitize_tool_name=_sanitize)
        tool_schema = next(s for s in schemas if s["function"]["name"] == "search")

        properties = tool_schema["function"]["parameters"]["properties"]
        required = tool_schema["function"]["parameters"]["required"]

        assert list(properties.keys())[0] == "thought"
        assert required[0] == "thought"

    def test_thought_is_first_property_for_zero_params_tool(self):
        class _Empty(BaseModel):
            pass

        tool = _make_tool("ping", _Empty)
        schemas = generate_function_calling_schemas(tools=[tool], delegation_allowed=False, sanitize_tool_name=_sanitize)
        tool_schema = next(s for s in schemas if s["function"]["name"] == "ping")

        properties = tool_schema["function"]["parameters"]["properties"]
        required = tool_schema["function"]["parameters"]["required"]

        assert list(properties.keys()) == ["thought"]
        assert required == ["thought"]

    def test_no_action_input_wrapper_in_any_schema(self):
        class _Schema(BaseModel):
            x: str = Field(..., description="X")

        tool = _make_tool("foo", _Schema)
        schemas = generate_function_calling_schemas(tools=[tool], delegation_allowed=False, sanitize_tool_name=_sanitize)

        for schema in schemas:
            properties = schema["function"]["parameters"]["properties"]
            assert "action_input" not in properties, f"action_input wrapper leaked into {schema['function']['name']}"

    def test_strict_when_all_params_required(self):
        class _Schema(BaseModel):
            a: str = Field(..., description="A")
            b: str = Field(..., description="B")

        tool = _make_tool("op", _Schema)
        schemas = generate_function_calling_schemas(tools=[tool], delegation_allowed=False, sanitize_tool_name=_sanitize)
        tool_schema = next(s for s in schemas if s["function"]["name"] == "op")

        assert tool_schema["function"]["strict"] is True
        assert tool_schema["function"]["parameters"]["additionalProperties"] is False
        assert set(tool_schema["function"]["parameters"]["required"]) == {"thought", "a", "b"}

    def test_strict_dropped_when_any_param_is_optional(self):
        class _Schema(BaseModel):
            required_field: str = Field(..., description="Required")
            optional_field: Optional[str] = Field(default=None, description="Optional")

        tool = _make_tool("op", _Schema)
        schemas = generate_function_calling_schemas(tools=[tool], delegation_allowed=False, sanitize_tool_name=_sanitize)
        tool_schema = next(s for s in schemas if s["function"]["name"] == "op")

        assert tool_schema["function"]["strict"] is False
        # required list excludes the optional field but always starts with thought
        assert tool_schema["function"]["parameters"]["required"][0] == "thought"
        assert "required_field" in tool_schema["function"]["parameters"]["required"]
        assert "optional_field" not in tool_schema["function"]["parameters"]["required"]

    def test_additional_properties_false_on_tool_schemas(self):
        """Flat-args tool schemas must set additionalProperties: false at the outer
        parameters level so strict mode (when enabled) blocks malformed shapes."""

        class _Schema(BaseModel):
            x: str = Field(..., description="X")

        tool = _make_tool("foo", _Schema)
        schemas = generate_function_calling_schemas(tools=[tool], delegation_allowed=False, sanitize_tool_name=_sanitize)

        for schema in schemas:
            if schema["function"]["name"] == "provide_final_answer":
                continue  # final-answer schema is untouched by this migration
            params = schema["function"]["parameters"]
            assert params.get("additionalProperties") is False, (
                f"{schema['function']['name']} missing additionalProperties: false at parameters level"
            )

    def test_zero_param_tool_is_strict_by_default(self):
        """A tool with no params produces a fully-required, strict schema."""

        class _Empty(BaseModel):
            pass

        tool = _make_tool("ping", _Empty)
        schemas = generate_function_calling_schemas(tools=[tool], delegation_allowed=False, sanitize_tool_name=_sanitize)
        tool_schema = next(s for s in schemas if s["function"]["name"] == "ping")

        assert tool_schema["function"]["strict"] is True


class TestFinalAnswerSchema:
    def test_final_answer_is_first_in_list(self):
        class _Schema(BaseModel):
            x: str = Field(..., description="X")

        tool = _make_tool("foo", _Schema)
        schemas = generate_function_calling_schemas(tools=[tool], delegation_allowed=False, sanitize_tool_name=_sanitize)

        assert schemas[0]["function"]["name"] == "provide_final_answer"

    def test_final_answer_keeps_thought_first(self):
        tool = _make_tool("foo", None)
        schemas = generate_function_calling_schemas(tools=[tool], delegation_allowed=False, sanitize_tool_name=_sanitize)

        properties = schemas[0]["function"]["parameters"]["properties"]
        assert list(properties.keys())[0] == "thought"


@pytest.fixture
def _sub_agent_tool():
    """Build a SubAgentTool-like mock for delegate_final tests.

    Uses ``spec=SubAgentTool`` so the ``isinstance(tool, SubAgentTool)`` branch in
    ``generate_function_calling_schemas`` is taken without instantiating the real
    Pydantic model (which has ``input_schema`` as a ClassVar).
    """
    from dynamiq.nodes.tools.agent_tool import SubAgentTool

    class _Schema(BaseModel):
        input: str = Field(..., description="Subtask")

    tool = MagicMock(spec=SubAgentTool)
    tool.name = "Researcher"
    tool.description = "Research tool"
    tool.input_schema = _Schema
    return tool


class TestSubAgentDelegateFinal:
    def test_delegate_final_is_top_level_sibling_not_nested(self, _sub_agent_tool):
        """In flat-args mode, `delegate_final` lives at the top level alongside the
        agent tool's own params (e.g. `input`), not inside an `action_input` wrapper."""
        schemas = generate_function_calling_schemas(
            tools=[_sub_agent_tool],
            delegation_allowed=True,
            sanitize_tool_name=_sanitize,
        )
        tool_schema = next(s for s in schemas if s["function"]["name"] == "Researcher")
        properties = tool_schema["function"]["parameters"]["properties"]

        assert "delegate_final" in properties
        assert "action_input" not in properties
        # thought is still first
        assert list(properties.keys())[0] == "thought"
