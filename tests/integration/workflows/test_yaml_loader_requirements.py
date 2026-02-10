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

EXTERNAL_USER_ID = "user-abc-123"
ACCOUNT_ID = "acc-456"
SECRET_TOKEN = "secret-token-123"
REFRESH_TOKEN = "refresh-xyz"
EXTRACTED_NAME = "extracted-name"
CONN_ID_RESOLVED = "conn-id-123"

VP_EXTERNAL_USER_ID = "$.external_user_id"
VP_ACCOUNT_ID = "$.account_id"
VP_CREDENTIALS_TOKEN = "$.credentials.token"

RESOLVED_USER_REQUIREMENT = {"external_user_id": EXTERNAL_USER_ID, "account_id": ACCOUNT_ID}


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
    resolved_data = {"type": OPENAI_CONN, "api_key": API_KEY, "id": CONN_ID_RESOLVED}
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


# --- value_path tests ---


def test_requirement_data_with_value_path():
    """Verify RequirementData captures value_path when present."""
    req = RequirementData.model_validate({"$type": "requirement", "$id": REQ_1, "value_path": VP_ACCOUNT_ID})

    assert req.type == "requirement"
    assert req.id == REQ_1
    assert req.value_path == VP_ACCOUNT_ID


def test_requirement_data_without_value_path():
    """Verify RequirementData defaults value_path to None when absent."""
    req = RequirementData.model_validate({"$type": "connection", "$id": REQ_1})

    assert req.value_path is None


def test_get_requirements_extracts_value_path():
    """Verify get_requirements captures value_path from requirement dicts."""
    data = {
        "nodes": {
            NODE_1: {
                "external_user_id": {"$type": "requirement", "$id": REQ_1, "value_path": VP_EXTERNAL_USER_ID},
            }
        }
    }
    requirements = WorkflowYAMLLoader.get_requirements(data)

    assert len(requirements) == 1
    assert requirements[0].id == REQ_1
    assert requirements[0].value_path == VP_EXTERNAL_USER_ID


def test_apply_resolved_with_value_path_extracts_field():
    """Verify value_path extracts a specific field from the resolved dict."""
    data = {
        "nodes": {
            NODE_1: {
                "external_user_id": {"$type": "requirement", "$id": REQ_1, "value_path": VP_EXTERNAL_USER_ID},
            }
        }
    }

    WorkflowYAMLLoader.apply_resolved_requirements(data, {REQ_1: RESOLVED_USER_REQUIREMENT})

    assert data["nodes"][NODE_1]["external_user_id"] == EXTERNAL_USER_ID


def test_apply_resolved_with_value_path_nested_field():
    """Verify value_path works with nested JSON paths."""
    data = {
        "nodes": {
            NODE_1: {
                "token": {"$type": "requirement", "$id": REQ_1, "value_path": VP_CREDENTIALS_TOKEN},
            }
        }
    }
    resolved_value = {"credentials": {"token": SECRET_TOKEN, "refresh_token": REFRESH_TOKEN}}

    WorkflowYAMLLoader.apply_resolved_requirements(data, {REQ_1: resolved_value})

    assert data["nodes"][NODE_1]["token"] == SECRET_TOKEN


def test_apply_resolved_same_id_different_value_paths():
    """Verify same $id with different value_paths extracts different values."""
    data = {
        "nodes": {
            NODE_1: {
                "external_user_id": {"$type": "requirement", "$id": REQ_1, "value_path": VP_EXTERNAL_USER_ID},
                "configurable_props": {
                    "auth_provision_id": {"$type": "requirement", "$id": REQ_1, "value_path": VP_ACCOUNT_ID},
                },
            }
        }
    }

    WorkflowYAMLLoader.apply_resolved_requirements(data, {REQ_1: RESOLVED_USER_REQUIREMENT})

    assert data["nodes"][NODE_1]["external_user_id"] == EXTERNAL_USER_ID
    assert data["nodes"][NODE_1]["configurable_props"]["auth_provision_id"] == ACCOUNT_ID


def test_apply_resolved_value_path_no_match_raises():
    """Verify value_path that doesn't match raises exception."""
    data = {
        "nodes": {
            NODE_1: {
                "field": {"$type": "requirement", "$id": REQ_1, "value_path": "$.nonexistent"},
            }
        }
    }

    with pytest.raises(WorkflowYAMLLoaderException, match="value_path"):
        WorkflowYAMLLoader.apply_resolved_requirements(data, {REQ_1: {"other_field": "value"}})


def test_apply_resolved_value_path_on_non_dict_raises():
    """Verify value_path on a non-dict/list resolved value raises exception."""
    data = {
        "nodes": {
            NODE_1: {
                "field": {"$type": "requirement", "$id": REQ_1, "value_path": "$.key"},
            }
        }
    }

    with pytest.raises(WorkflowYAMLLoaderException, match="must be a dict or list"):
        WorkflowYAMLLoader.apply_resolved_requirements(data, {REQ_1: "plain-string"})


def test_apply_resolved_value_path_multiple_matches_raises():
    """Verify value_path that matches multiple values raises exception."""
    data = {
        "nodes": {
            NODE_1: {
                "field": {"$type": "requirement", "$id": REQ_1, "value_path": "$.items[*].id"},
            }
        }
    }
    resolved_value = {"items": [{"id": "a"}, {"id": "b"}, {"id": "c"}]}

    with pytest.raises(WorkflowYAMLLoaderException, match="matched multiple values"):
        WorkflowYAMLLoader.apply_resolved_requirements(data, {REQ_1: resolved_value})


def test_apply_resolved_value_path_in_list():
    """Verify value_path works for requirements inside lists."""
    data = {"items": [{"$type": "requirement", "$id": REQ_1, "value_path": "$.name"}]}

    WorkflowYAMLLoader.apply_resolved_requirements(data, {REQ_1: {"name": EXTRACTED_NAME, "age": 30}})

    assert data["items"] == [EXTRACTED_NAME]


def test_apply_resolved_mixed_with_and_without_value_path():
    """Verify requirements with and without value_path coexist correctly."""
    data = {
        "nodes": {
            NODE_1: {
                "connection": {"$type": "connection", "$id": REQ_1},
                "user_id": {"$type": "requirement", "$id": REQ_2, "value_path": VP_EXTERNAL_USER_ID},
            }
        }
    }
    resolved_connection = {"type": OPENAI_CONN, "api_key": API_KEY, "id": CONN_ID_RESOLVED}

    WorkflowYAMLLoader.apply_resolved_requirements(data, {REQ_1: resolved_connection, REQ_2: RESOLVED_USER_REQUIREMENT})

    assert data["nodes"][NODE_1]["connection"] == resolved_connection
    assert data["nodes"][NODE_1]["user_id"] == EXTERNAL_USER_ID
