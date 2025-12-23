import pytest

from dynamiq.serializers.loaders.yaml import WorkflowYAMLLoader, WorkflowYAMLLoaderException
from dynamiq.serializers.types import ObjectType, RequirementData

REQUIREMENT = ObjectType.REQUIREMENT.value

OPENAI_NODE = "dynamiq.nodes.llms.OpenAI"
ANTHROPIC_NODE = "dynamiq.nodes.llms.Anthropic"
OPENAI_CONN = "dynamiq.connections.OpenAI"
ANTHROPIC_CONN = "dynamiq.connections.Anthropic"

NODE_1 = "node1"
NODE_2 = "node2"
CONN_1 = "conn-1"
CONN_REGULAR = "regular"
CONN_EXPLICIT = "explicit"

REQ_1 = "req-1"
REQ_2 = "req-2"
REQ_IGNORED = "ignored"
REQ_MISSING = "missing"
REQ_OTHER = "other"

API_KEY = "test-api-key"
CUSTOM_URL = "https://custom.api.com"
RESOLVED_URL = "https://resolved.api.com"

MODEL_GPT = "gpt-4o"
MODEL_CLAUDE = "claude-4-sonnet"


@pytest.fixture
def single_requirement_data():
    return {
        "nodes": {
            NODE_1: {
                "type": OPENAI_NODE,
                "connection": {"type": OPENAI_CONN, "object": REQUIREMENT, "requirement_id": REQ_1},
            }
        }
    }


@pytest.fixture
def multiple_requirements_data():
    return {
        "nodes": {
            NODE_1: {
                "type": OPENAI_NODE,
                "connection": {"type": OPENAI_CONN, "object": REQUIREMENT, "requirement_id": REQ_1},
            },
            NODE_2: {
                "type": ANTHROPIC_NODE,
                "connection": {"type": ANTHROPIC_CONN, "object": REQUIREMENT, "requirement_id": REQ_2},
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
                "schema": {"object": REQUIREMENT, "requirement_id": REQ_IGNORED},
                "response_format": {"object": REQUIREMENT, "requirement_id": REQ_IGNORED},
                "connection": {"type": OPENAI_CONN, "object": REQUIREMENT, "requirement_id": REQ_1},
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
                "connection": {"type": OPENAI_CONN, "object": REQUIREMENT, "requirement_id": REQ_1},
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
            NODE_1: {"type": ANTHROPIC_NODE, "model": MODEL_CLAUDE, "connection": CONN_REGULAR},
            NODE_2: {
                "type": OPENAI_NODE,
                "model": MODEL_GPT,
                "connection": {"type": OPENAI_CONN, "object": REQUIREMENT, "requirement_id": REQ_1},
            },
        },
        "flows": {},
        "workflows": {},
    }


def test_get_requirements_extracts_single(single_requirement_data):
    requirements = WorkflowYAMLLoader.get_requirements(single_requirement_data)

    assert len(requirements) == 1
    assert requirements[0].requirement_id == REQ_1
    assert requirements[0].object == REQUIREMENT


def test_get_requirements_extracts_multiple(multiple_requirements_data):
    requirements = WorkflowYAMLLoader.get_requirements(multiple_requirements_data)

    assert len(requirements) == 2
    assert {r.requirement_id for r in requirements} == {REQ_1, REQ_2}


def test_get_requirements_returns_empty_when_none(regular_connection_data):
    assert len(WorkflowYAMLLoader.get_requirements(regular_connection_data)) == 0


def test_get_requirements_ignores_schema_fields(schema_fields_data):
    requirements = WorkflowYAMLLoader.get_requirements(schema_fields_data)

    assert len(requirements) == 1
    assert requirements[0].requirement_id == REQ_1


def test_apply_resolved_merges_data(single_requirement_data):
    WorkflowYAMLLoader.apply_resolved_requirements(single_requirement_data, {REQ_1: {"api_key": API_KEY}})

    connection = single_requirement_data["nodes"][NODE_1]["connection"]
    assert connection["api_key"] == API_KEY
    assert connection["requirement_id"] == REQ_1


def test_apply_resolved_does_not_overwrite_by_default():
    data = {
        "nodes": {
            NODE_1: {
                "connection": {
                    "type": OPENAI_CONN,
                    "object": REQUIREMENT,
                    "requirement_id": REQ_1,
                    "url": CUSTOM_URL,
                }
            }
        }
    }

    WorkflowYAMLLoader.apply_resolved_requirements(data, {REQ_1: {"api_key": API_KEY, "url": RESOLVED_URL}})

    connection = data["nodes"][NODE_1]["connection"]
    assert connection["api_key"] == API_KEY
    assert connection["url"] == CUSTOM_URL


def test_apply_resolved_overwrites_when_enabled():
    data = {
        "nodes": {
            NODE_1: {
                "connection": {
                    "type": OPENAI_CONN,
                    "object": REQUIREMENT,
                    "requirement_id": REQ_1,
                    "url": CUSTOM_URL,
                }
            }
        }
    }

    WorkflowYAMLLoader.apply_resolved_requirements(
        data, {REQ_1: {"api_key": API_KEY, "url": RESOLVED_URL}}, overwrite=True
    )

    connection = data["nodes"][NODE_1]["connection"]
    assert connection["api_key"] == API_KEY
    assert connection["url"] == RESOLVED_URL


def test_apply_resolved_raises_on_missing():
    data = {
        "nodes": {NODE_1: {"connection": {"type": OPENAI_CONN, "object": REQUIREMENT, "requirement_id": REQ_MISSING}}}
    }

    with pytest.raises(WorkflowYAMLLoaderException) as exc_info:
        WorkflowYAMLLoader.apply_resolved_requirements(data, {REQ_OTHER: {"api_key": API_KEY}})

    assert REQ_MISSING in str(exc_info.value)


def test_apply_resolved_handles_empty():
    data = {"nodes": {NODE_1: {"type": OPENAI_NODE}}}
    original = {"nodes": {NODE_1: {"type": OPENAI_NODE}}}

    WorkflowYAMLLoader.apply_resolved_requirements(data, {})

    assert data == original


def test_requirement_data_from_valid_dict():
    result = RequirementData.from_dict({"object": REQUIREMENT, "requirement_id": REQ_1})

    assert result is not None
    assert result.requirement_id == REQ_1


def test_requirement_data_from_dict_wrong_object():
    assert RequirementData.from_dict({"object": REQ_OTHER, "requirement_id": REQ_1}) is None


def test_requirement_data_from_dict_missing_object():
    assert RequirementData.from_dict({"requirement_id": REQ_1}) is None


def test_requirement_data_from_dict_missing_id():
    assert RequirementData.from_dict({"object": REQUIREMENT}) is None


def test_inline_connection_uses_requirement_id(parseable_data):
    WorkflowYAMLLoader.apply_resolved_requirements(parseable_data, {REQ_1: {"api_key": API_KEY}})
    result = WorkflowYAMLLoader.parse(parseable_data, init_components=False)

    assert REQ_1 in result.connections
    assert result.connections[REQ_1].id == REQ_1


def test_inline_connection_respects_explicit_id():
    data = {
        "connections": {},
        "nodes": {
            NODE_1: {
                "type": OPENAI_NODE,
                "model": MODEL_GPT,
                "connection": {
                    "id": CONN_EXPLICIT,
                    "type": OPENAI_CONN,
                    "object": REQUIREMENT,
                    "requirement_id": REQ_1,
                },
            }
        },
        "flows": {},
        "workflows": {},
    }

    WorkflowYAMLLoader.apply_resolved_requirements(data, {REQ_1: {"api_key": API_KEY}})
    result = WorkflowYAMLLoader.parse(data, init_components=False)

    assert CONN_EXPLICIT in result.connections


def test_mixed_connections_work_together(mixed_connections_data):
    WorkflowYAMLLoader.apply_resolved_requirements(mixed_connections_data, {REQ_1: {"api_key": API_KEY}})
    result = WorkflowYAMLLoader.parse(mixed_connections_data, init_components=False)

    assert len(result.connections) == 2
    assert CONN_REGULAR in result.connections
    assert REQ_1 in result.connections
