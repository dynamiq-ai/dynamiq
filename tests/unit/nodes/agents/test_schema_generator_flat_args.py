"""Tests for the flat-args function-calling schema shape.

Verifies that the schema generator produces schemas where:
  * `thought` is the FIRST property (load-bearing for streaming UX and model
    chain-of-thought behavior).
  * Tool params are TOP-LEVEL siblings of `thought` (no `action_input` wrapper).
  * `additionalProperties: false` is set on `parameters`.
  * Only genuinely-required fields go in `required`. Strict mode (and its
    all-required promotion) is applied per-provider in
    `BaseLLM.transform_tool_schemas`, not at generation time, so no `strict`
    key is emitted here.
"""

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

from dynamiq.nodes.agents.components.schema_generator import generate_function_calling_schemas


def _sanitize(name: str) -> str:
    return name


def _gen(tool, delegation_allowed: bool = False):
    return generate_function_calling_schemas(
        tools=[tool], delegation_allowed=delegation_allowed, sanitize_tool_name=_sanitize
    )


def _make_tool(name: str, input_schema_cls: type[BaseModel] | None = None, description: str = "desc"):
    """Build a mock tool with a Pydantic input schema."""

    class _NoFields(BaseModel):
        pass

    tool = MagicMock()
    tool.name = name
    tool.description = description
    schema = input_schema_cls if input_schema_cls is not None else _NoFields
    tool.input_schema = schema
    # Mirror a real Node: resolved_input_schema returns input_schema when no
    # input_param_modes are set. The generator reads resolved_input_schema.
    tool.resolved_input_schema = schema
    return tool


class _RequiredOnlySchema(BaseModel):
    file_path: str = Field(..., description="Path")
    content: str = Field(..., description="Content")


class _MixedParamsSchema(BaseModel):
    query: str = Field(..., description="Query")
    limit: int = Field(default=10, description="Max results")


class _EmptySchema(BaseModel):
    pass


class TestFlatArgsSchema:
    @pytest.mark.parametrize(
        "schema_cls, expected_properties",
        [
            (_RequiredOnlySchema, ["thought", "file_path", "content"]),
            (_MixedParamsSchema, ["thought", "query", "limit"]),
            (_EmptySchema, ["thought"]),
        ],
        ids=["required_only", "mixed_params", "zero_params"],
    )
    def test_thought_is_first_and_params_are_flat_siblings(self, schema_cls, expected_properties):
        """`thought` is the first property and tool params are top-level siblings,
        regardless of whether params are all-required, mixed, or absent."""
        tool = _make_tool("tool", schema_cls)
        tool_schema = next(s for s in _gen(tool) if s["function"]["name"] == "tool")

        properties = tool_schema["function"]["parameters"]["properties"]
        required = tool_schema["function"]["parameters"]["required"]

        assert list(properties.keys()) == expected_properties
        assert required[0] == "thought"

    def test_no_action_input_wrapper_in_any_schema(self):
        class _Schema(BaseModel):
            x: str = Field(..., description="X")

        tool = _make_tool("foo", _Schema)
        schemas = _gen(tool)

        for schema in schemas:
            properties = schema["function"]["parameters"]["properties"]
            assert "action_input" not in properties, f"action_input wrapper leaked into {schema['function']['name']}"

    def test_all_required_params_in_required_list(self):
        class _Schema(BaseModel):
            a: str = Field(..., description="A")
            b: str = Field(..., description="B")

        tool = _make_tool("op", _Schema)
        schemas = _gen(tool)
        tool_schema = next(s for s in schemas if s["function"]["name"] == "op")

        # No gen-time strict key — strict is applied per-provider downstream.
        assert "strict" not in tool_schema["function"]
        assert tool_schema["function"]["parameters"]["additionalProperties"] is False
        assert set(tool_schema["function"]["parameters"]["required"]) == {"thought", "a", "b"}

    def test_optional_params_excluded_from_required_list(self):
        class _Schema(BaseModel):
            required_field: str = Field(..., description="Required")
            optional_field: str | None = Field(default=None, description="Optional")

        tool = _make_tool("op", _Schema)
        schemas = _gen(tool)
        tool_schema = next(s for s in schemas if s["function"]["name"] == "op")

        assert "strict" not in tool_schema["function"]
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
        schemas = _gen(tool)

        for schema in schemas:
            if schema["function"]["name"] == "provide_final_answer":
                continue  # final-answer schema is untouched by this migration
            params = schema["function"]["parameters"]
            assert (
                params.get("additionalProperties") is False
            ), f"{schema['function']['name']} missing additionalProperties: false at parameters level"

    def test_zero_param_tool_requires_only_thought(self):
        """A tool with no params produces a closed schema requiring only `thought`."""

        class _Empty(BaseModel):
            pass

        tool = _make_tool("ping", _Empty)
        schemas = _gen(tool)
        tool_schema = next(s for s in schemas if s["function"]["name"] == "ping")
        params = tool_schema["function"]["parameters"]

        assert "strict" not in tool_schema["function"]
        assert params["additionalProperties"] is False
        assert params["required"] == ["thought"]

    def test_extra_allow_tool_is_open(self):
        """A no-declared-fields tool that accepts extras (e.g. the generic Python
        tool) must stay OPEN: additionalProperties true, so the model can pass
        arbitrary params as top-level siblings of `thought`."""
        from pydantic import ConfigDict

        class _Dynamic(BaseModel):
            model_config = ConfigDict(extra="allow")

        tool = _make_tool("run_code", _Dynamic)
        schemas = _gen(tool)
        tool_schema = next(s for s in schemas if s["function"]["name"] == "run_code")
        params = tool_schema["function"]["parameters"]

        assert params["additionalProperties"] is True
        assert "strict" not in tool_schema["function"]
        assert params["required"] == ["thought"]


class TestFinalAnswerSchema:
    def test_final_answer_is_first_in_list(self):
        class _Schema(BaseModel):
            x: str = Field(..., description="X")

        tool = _make_tool("foo", _Schema)
        schemas = _gen(tool)

        assert schemas[0]["function"]["name"] == "provide_final_answer"

    def test_final_answer_keeps_thought_first(self):
        tool = _make_tool("foo", None)
        schemas = _gen(tool)

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
    # The generator reads resolved_input_schema (mirrors input_schema with no param modes).
    tool.resolved_input_schema = _Schema
    return tool


class TestSubAgentDelegateFinal:
    def test_delegate_final_is_top_level_sibling_not_nested(self, _sub_agent_tool):
        """In flat-args mode, `delegate_final` lives at the top level alongside the
        agent tool's own params (e.g. `input`), not inside an `action_input` wrapper."""
        schemas = _gen(_sub_agent_tool, delegation_allowed=True)
        tool_schema = next(s for s in schemas if s["function"]["name"] == "Researcher")
        properties = tool_schema["function"]["parameters"]["properties"]

        assert "delegate_final" in properties
        assert "action_input" not in properties
        # thought is still first
        assert list(properties.keys())[0] == "thought"
