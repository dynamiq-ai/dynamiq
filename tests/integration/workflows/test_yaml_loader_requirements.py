import pytest

from dynamiq.serializers.loaders.yaml import WorkflowYAMLLoader, WorkflowYAMLLoaderException
from dynamiq.serializers.types import RequirementData

OPENAI_NODE = "dynamiq.nodes.llms.OpenAI"
ANTHROPIC_NODE = "dynamiq.nodes.llms.Anthropic"
OPENAI_CONN = "dynamiq.connections.OpenAI"
ANTHROPIC_CONN = "dynamiq.connections.Anthropic"

NODE_1 = "node1"
NODE_2 = "node2"
CONN_1 = "conn-1"
CONN_REGULAR = "regular"

REQ_1 = "req-1"
REQ_2 = "req-2"
REQ_IGNORED = "ignored"
REQ_MISSING = "missing"
REQ_OTHER = "other"

API_KEY = "test-api-key"
MODEL_GPT = "gpt-4o"


@pytest.fixture
def single_requirement_data():
    return {
        "nodes": {
            NODE_1: {
                "type": OPENAI_NODE,
                "connection": {"$type": "connection", "$id": REQ_1},
            }
        }
    }


@pytest.fixture
def multiple_requirements_data():
    return {
        "nodes": {
            NODE_1: {
                "type": OPENAI_NODE,
                "connection": {"$type": "connection", "$id": REQ_1},
            },
            NODE_2: {
                "type": ANTHROPIC_NODE,
                "connection": {"$type": "connection", "$id": REQ_2},
            },
        }
    }


@pytest.fixture
def regular_connection_data():
    return {
        "connections": {CONN_1: {"type": OPENAI_CONN, "api_key": API_KEY}},
        "nodes": {NODE_1: {"type": OPENAI_NODE, "connection": CONN_1}},
    }


@pytest.fixture
def schema_fields_data():
    return {
        "nodes": {
            NODE_1: {
                "type": OPENAI_NODE,
                "schema": {"$type": "schema", "$id": REQ_IGNORED},
                "response_format": {"$type": "format", "$id": REQ_IGNORED},
                "connection": {"$type": "connection", "$id": REQ_1},
            }
        }
    }


@pytest.fixture
def parseable_data():
    return {
        "connections": {},
        "nodes": {
            NODE_1: {
                "type": OPENAI_NODE,
                "model": MODEL_GPT,
                "connection": {"$type": "connection", "$id": REQ_1},
            }
        },
        "flows": {},
        "workflows": {},
    }


@pytest.fixture
def mixed_connections_data():
    return {
        "connections": {CONN_REGULAR: {"type": ANTHROPIC_CONN, "api_key": API_KEY}},
        "nodes": {
            NODE_1: {"type": ANTHROPIC_NODE, "model": "claude-4-sonnet", "connection": CONN_REGULAR},
            NODE_2: {
                "type": OPENAI_NODE,
                "model": MODEL_GPT,
                "connection": {"$type": "connection", "$id": REQ_1},
            },
        },
        "flows": {},
        "workflows": {},
    }


def test_get_requirements_extracts_single(single_requirement_data):
    requirements = WorkflowYAMLLoader.get_requirements(single_requirement_data)

    assert len(requirements) == 1
    assert requirements[0].id == REQ_1
    assert requirements[0].type == "connection"


def test_get_requirements_extracts_multiple(multiple_requirements_data):
    requirements = WorkflowYAMLLoader.get_requirements(multiple_requirements_data)

    assert len(requirements) == 2
    assert {r.id for r in requirements} == {REQ_1, REQ_2}


def test_get_requirements_returns_empty_when_none(regular_connection_data):
    assert WorkflowYAMLLoader.get_requirements(regular_connection_data) == []


def test_get_requirements_ignores_schema_fields(schema_fields_data):
    """Verify that schema and response_format fields are skipped."""
    requirements = WorkflowYAMLLoader.get_requirements(schema_fields_data)

    assert len(requirements) == 1
    assert requirements[0].id == REQ_1


def test_get_requirements_requires_both_type_and_id():
    """Verify that both $type and $id are required to identify a requirement."""
    data_with_only_id = {"nodes": {NODE_1: {"connection": {"$id": REQ_1}}}}
    data_with_only_type = {"nodes": {NODE_1: {"connection": {"$type": "connection"}}}}
    data_with_both = {"nodes": {NODE_1: {"connection": {"$type": "connection", "$id": REQ_1}}}}

    assert WorkflowYAMLLoader.get_requirements(data_with_only_id) == []
    assert WorkflowYAMLLoader.get_requirements(data_with_only_type) == []

    result = WorkflowYAMLLoader.get_requirements(data_with_both)
    assert len(result) == 1
    assert result[0].id == REQ_1
    assert result[0].type == "connection"


def test_requirement_data_model():
    """Verify RequirementData model works correctly."""
    req = RequirementData.model_validate({"$type": "connection", "$id": REQ_1})

    assert req.type == "connection"
    assert req.id == REQ_1


def test_requirement_data_from_dict():
    """Verify RequirementData.from_dict works correctly."""
    result = RequirementData.from_dict({"$type": "connection", "$id": REQ_1})

    assert result is not None
    assert result.type == "connection"
    assert result.id == REQ_1


def test_requirement_data_from_dict_returns_none_without_type():
    assert RequirementData.from_dict({"$id": REQ_1}) is None


def test_requirement_data_from_dict_returns_none_without_id():
    assert RequirementData.from_dict({"$type": "connection"}) is None


def test_requirement_data_from_dict_returns_none_for_empty():
    assert RequirementData.from_dict({}) is None


def test_apply_resolved_replaces_requirement_dict(single_requirement_data):
    """Verify that requirement dict is completely replaced with resolved data."""
    resolved_data = {"type": OPENAI_CONN, "api_key": API_KEY, "id": "conn-id-123"}
    WorkflowYAMLLoader.apply_resolved_requirements(single_requirement_data, {REQ_1: resolved_data})

    connection = single_requirement_data["nodes"][NODE_1]["connection"]
    assert connection == resolved_data
    assert "$id" not in connection
    assert "$type" not in connection


def test_apply_resolved_replaces_entire_dict():
    """Verify that original fields in requirement dict are removed after resolution."""
    data = {
        "nodes": {
            NODE_1: {
                "connection": {
                    "$type": "connection",
                    "$id": REQ_1,
                    "url": "https://custom.api.com",
                }
            }
        }
    }
    resolved_url = "https://resolved.api.com"

    resolved_data = {"type": OPENAI_CONN, "api_key": API_KEY, "url": resolved_url}
    WorkflowYAMLLoader.apply_resolved_requirements(data, {REQ_1: resolved_data})

    connection = data["nodes"][NODE_1]["connection"]
    assert connection == resolved_data
    assert connection["url"] == resolved_url


def test_apply_resolved_raises_on_missing():
    data = {"nodes": {NODE_1: {"connection": {"$type": "connection", "$id": REQ_MISSING}}}}

    with pytest.raises(WorkflowYAMLLoaderException) as exc_info:
        WorkflowYAMLLoader.apply_resolved_requirements(data, {REQ_OTHER: {"api_key": API_KEY}})

    assert REQ_MISSING in str(exc_info.value)


def test_apply_resolved_handles_empty_data_no_requirements():
    data = {"nodes": {NODE_1: {"type": OPENAI_NODE}}}
    original = {"nodes": {NODE_1: {"type": OPENAI_NODE}}}

    WorkflowYAMLLoader.apply_resolved_requirements(data, {})

    assert data == original


def test_apply_resolved_raises_on_empty_resolved_with_requirements():
    """Verify that empty resolved_requirements raises when data contains requirements."""
    data = {"nodes": {NODE_1: {"connection": {"$type": "connection", "$id": REQ_1}}}}

    with pytest.raises(WorkflowYAMLLoaderException) as exc_info:
        WorkflowYAMLLoader.apply_resolved_requirements(data, {})

    assert REQ_1 in str(exc_info.value)


def test_apply_resolved_ignores_dict_with_only_id():
    """Verify that dicts with only $id (no $type) are not treated as requirements."""
    data = {"nodes": {NODE_1: {"connection": {"$id": REQ_1, "some_field": "value"}}}}
    original_connection = {"$id": REQ_1, "some_field": "value"}

    WorkflowYAMLLoader.apply_resolved_requirements(data, {REQ_1: {"api_key": API_KEY}})

    # Should remain unchanged since $type is missing
    assert data["nodes"][NODE_1]["connection"] == original_connection


def test_inline_connection_with_resolved_requirement(parseable_data):
    """Verify that resolved requirement provides complete connection data."""
    resolved_connection = {"type": OPENAI_CONN, "api_key": API_KEY, "id": "resolved-conn-id"}
    WorkflowYAMLLoader.apply_resolved_requirements(parseable_data, {REQ_1: resolved_connection})
    result = WorkflowYAMLLoader.parse(parseable_data, init_components=False)

    assert "resolved-conn-id" in result.connections
    assert result.connections["resolved-conn-id"].id == "resolved-conn-id"


def test_inline_connection_generates_uuid_if_no_id():
    """Verify that a UUID is generated for inline connection if id is not provided."""
    data = {
        "connections": {},
        "nodes": {
            NODE_1: {
                "type": OPENAI_NODE,
                "model": MODEL_GPT,
                "connection": {"$type": "connection", "$id": REQ_1},
            }
        },
        "flows": {},
        "workflows": {},
    }

    resolved_connection = {"type": OPENAI_CONN, "api_key": API_KEY}  # No id provided
    WorkflowYAMLLoader.apply_resolved_requirements(data, {REQ_1: resolved_connection})
    result = WorkflowYAMLLoader.parse(data, init_components=False)

    # Should have exactly one connection with a generated UUID
    assert len(result.connections) == 1
    conn_id = list(result.connections.keys())[0]
    assert result.connections[conn_id].api_key == API_KEY


def test_mixed_connections_work_together(mixed_connections_data):
    """Verify that regular and inline connections work together."""
    resolved_connection = {"type": OPENAI_CONN, "api_key": API_KEY, "id": "resolved-conn"}
    WorkflowYAMLLoader.apply_resolved_requirements(mixed_connections_data, {REQ_1: resolved_connection})
    result = WorkflowYAMLLoader.parse(mixed_connections_data, init_components=False)

    assert len(result.connections) == 2
    assert CONN_REGULAR in result.connections
    assert "resolved-conn" in result.connections


def test_apply_resolved_with_string_value():
    """Verify that resolved value can be a string."""
    data = {"nodes": {NODE_1: {"api_key": {"$type": "secret", "$id": REQ_1}}}}
    secret_key = "my-secret-api-key"

    WorkflowYAMLLoader.apply_resolved_requirements(data, {REQ_1: secret_key})

    assert data["nodes"][NODE_1]["api_key"] == secret_key


def test_apply_resolved_with_list_value():
    """Verify that resolved value can be a list."""
    data = {"nodes": {NODE_1: {"tags": {"$type": "tags", "$id": REQ_1}}}}
    tags = ["tag1", "tag2", "tag3"]

    WorkflowYAMLLoader.apply_resolved_requirements(data, {REQ_1: tags})

    assert data["nodes"][NODE_1]["tags"] == tags


def test_apply_resolved_with_integer_value():
    """Verify that resolved value can be an integer."""
    data = {"nodes": {NODE_1: {"max_tokens": {"$type": "config", "$id": REQ_1}}}}

    WorkflowYAMLLoader.apply_resolved_requirements(data, {REQ_1: 4096})

    assert data["nodes"][NODE_1]["max_tokens"] == 4096


def test_apply_resolved_with_boolean_value():
    """Verify that resolved value can be a boolean."""
    data = {"nodes": {NODE_1: {"stream": {"$type": "config", "$id": REQ_1}}}}

    WorkflowYAMLLoader.apply_resolved_requirements(data, {REQ_1: True})

    assert data["nodes"][NODE_1]["stream"] is True


def test_apply_resolved_with_none_value():
    """Verify that resolved value can be None."""
    data = {"nodes": {NODE_1: {"optional_field": {"$type": "config", "$id": REQ_1}}}}

    WorkflowYAMLLoader.apply_resolved_requirements(data, {REQ_1: None})

    assert data["nodes"][NODE_1]["optional_field"] is None


def test_apply_resolved_in_list_with_string_value():
    """Verify that requirements inside lists can be resolved to strings."""
    data = {"items": [{"$type": "item", "$id": REQ_1}, {"$type": "item", "$id": REQ_2}]}

    WorkflowYAMLLoader.apply_resolved_requirements(data, {REQ_1: "first", REQ_2: "second"})

    assert data["items"] == ["first", "second"]


def test_apply_resolved_in_list_raises_on_missing():
    """Verify that missing requirement in list raises exception."""
    data = {"items": [{"$type": "item", "$id": REQ_MISSING}]}

    with pytest.raises(WorkflowYAMLLoaderException) as exc_info:
        WorkflowYAMLLoader.apply_resolved_requirements(data, {})

    assert REQ_MISSING in str(exc_info.value)
