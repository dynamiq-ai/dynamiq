"""Unit tests for the Composio action tool.

A Composio execute answers HTTP 200 even when the action itself failed, so the tests below pin
both the transport-level and the body-level failure handling, and assert a single execute issues
a single request - an execute can have side effects.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dynamiq.connections.connections import Composio as ComposioConnection
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import Node
from dynamiq.nodes.tools.composio import Composio
from dynamiq.runnables import RunnableStatus

INPUT_PROPS = {
    "type": "object",
    "title": "SendEmailRequest",
    "required": ["recipient_email", "body"],
    "properties": {
        "recipient_email": {"type": "string", "description": "Address of the recipient."},
        "body": {"type": "string", "description": "Body of the email."},
        "subject": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "default": None,
            "description": "Subject of the email.",
        },
        "is_html": {"type": "boolean", "default": False, "description": "Whether the body is HTML."},
        "extra-header": {"type": "string", "description": "Header that is not a python identifier."},
    },
}


def _mock_response(status_code=200, json_payload=None, text="ok"):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.json = MagicMock(return_value=json_payload if json_payload is not None else {})
    return resp


NESTED_INPUT_PROPS = {
    "$defs": {
        "Req": {
            "type": "object",
            "required": ["addr"],
            "properties": {
                "addr": {"$ref": "#/$defs/Addr"},
                "stops": {"type": "array", "items": {"$ref": "#/$defs/Addr"}},
                "note": {"type": "string"},
            },
        },
        # More than one optional sub-field, so a test asserting the payload shape cannot pass
        # merely because there was nothing left to leak.
        "Addr": {
            "type": "object",
            "required": ["city"],
            "properties": {
                "city": {"type": "string"},
                "zip": {"type": "string"},
                "street": {"type": "string"},
            },
        },
    },
    "$ref": "#/$defs/Req",
}


def _composed_root(inner: dict, style: str) -> dict:
    """Express `inner` as a root that only reaches its properties through $ref or allOf."""
    if style == "inline":
        return inner
    if style == "$ref":
        return {"$defs": {"Req": inner}, "$ref": "#/$defs/Req"}
    return {"$defs": {"Req": inner}, "allOf": [{"$ref": "#/$defs/Req"}]}


COMPOSED_ROOT_STYLES = ["inline", "$ref", "allOf"]


def _build_node(**kwargs) -> Composio:
    kwargs.setdefault("input_props", INPUT_PROPS)
    kwargs.setdefault("user_id", "project-uuid")
    kwargs.setdefault("toolkit_slug", "gmail")
    kwargs.setdefault("tool_slug", "GMAIL_SEND_EMAIL")
    return Composio(connection=ComposioConnection(api_key="key"), **kwargs)


class TestComposioInputSchema:
    def test_required_props_are_required_and_the_rest_optional(self):
        node = _build_node()

        fields = node.input_schema.model_fields
        required = {name for name, field in fields.items() if field.is_required()}

        assert required == {"recipient_email", "body"}
        assert fields["subject"].default is None
        assert fields["is_html"].default is False

    def test_configured_argument_stops_being_required(self):
        node = _build_node(arguments={"body": "Configured body"})

        fields = node.input_schema.model_fields
        required = {name for name, field in fields.items() if field.is_required()}

        assert required == {"recipient_email"}

    @pytest.mark.parametrize("style", COMPOSED_ROOT_STYLES)
    def test_configured_argument_stops_being_required_on_a_composed_root(self, style):
        # Composio may express a tool's schema through a root-level $ref or allOf. The schema builder
        # resolves such a root before reading its properties, so the configured-argument handling has
        # to resolve it too - otherwise the agent is asked for a parameter it already has.
        inner = {
            "type": "object",
            "required": ["recipient", "body"],
            "properties": {"recipient": {"type": "string"}, "body": {"type": "string"}},
        }
        node = _build_node(
            input_props=_composed_root(inner, style),
            arguments={"recipient": "configured@example.com"},
        )

        required = {name for name, field in node.input_schema.model_fields.items() if field.is_required()}

        assert required == {"body"}

    @pytest.mark.parametrize("style", COMPOSED_ROOT_STYLES)
    def test_configured_argument_survives_the_schema_default_on_a_composed_root(self, style):
        # The silent case: no validation error, the action just executes against Composio's default
        # instead of the value configured at design time.
        inner = {
            "type": "object",
            "required": ["body"],
            "properties": {
                "recipient": {"type": "string", "default": "composio-default@example.com"},
                "body": {"type": "string"},
            },
        }
        node = _build_node(
            input_props=_composed_root(inner, style),
            arguments={"recipient": "configured@example.com"},
        )

        _, payload = node._build_request(node.input_schema(body="hi"))

        assert payload["arguments"]["recipient"] == "configured@example.com"

    def test_definitions_survive_a_resolved_root(self):
        # Resolving the root drops the `$defs` that lived on it, so they have to be carried across
        # or a property referencing one of them would fail to build.
        node = _build_node(input_props=NESTED_INPUT_PROPS)

        _, payload = node._build_request(node.input_schema(addr={"city": "Kyiv"}, note="x"))

        assert sorted(node.input_schema.model_fields) == ["addr", "note", "stops"]
        assert payload["arguments"]["addr"] == {"city": "Kyiv"}

    def test_unset_fields_are_dropped_at_every_depth(self):
        # The payload must carry only what the caller supplied: an optional property is typed
        # `X | None` with default None, so a nested object would otherwise post an explicit null
        # for every sub-field nobody set.
        node = _build_node(input_props=NESTED_INPUT_PROPS)

        _, payload = node._build_request(node.input_schema(addr={"city": "Kyiv"}, stops=[{"city": "Lviv"}]))

        assert payload["arguments"] == {"addr": {"city": "Kyiv"}, "stops": [{"city": "Lviv"}]}

    def test_supplied_nested_values_are_kept(self):
        node = _build_node(input_props=NESTED_INPUT_PROPS)

        _, payload = node._build_request(node.input_schema(addr={"city": "Kyiv", "zip": "01001"}, note="n"))

        assert payload["arguments"] == {"addr": {"city": "Kyiv", "zip": "01001"}, "note": "n"}

    def test_property_name_that_is_not_an_identifier_keeps_its_alias(self):
        node = _build_node()

        field = node.input_schema.model_fields["extra_header"]

        assert field.alias == "extra-header"

    @pytest.mark.parametrize("input_props", [{}, None])
    def test_undeclared_schema_builds_a_model_without_fields(self, input_props):
        # Composio reports "no declared schema" as either `{}` or `null`, and the persisted node omits
        # the key entirely in that case, so all three spellings must build an empty input schema.
        node = _build_node(input_props=input_props)

        assert node.input_schema.model_fields == {}

    def test_absent_input_props_builds_a_model_without_fields(self):
        node = Composio(
            connection=ComposioConnection(api_key="key"),
            user_id="project-uuid",
            toolkit_slug="gmail",
            tool_slug="GMAIL_SEND_EMAIL",
        )

        assert node.input_schema.model_fields == {}

    def test_node_type_matches_the_wire_contract(self):
        assert _build_node().type == "dynamiq.nodes.tools.Composio"

    def test_to_dict_carries_the_wire_contract_keys(self):
        node = _build_node(
            tool_version="20250905_00",
            connected_account_id="ca_xxx",
            arguments={"recipient_email": "a@b.c"},
        )

        dumped = node.to_dict()

        assert dumped["type"] == "dynamiq.nodes.tools.Composio"
        assert dumped["user_id"] == "project-uuid"
        assert dumped["toolkit_slug"] == "gmail"
        assert dumped["tool_slug"] == "GMAIL_SEND_EMAIL"
        assert dumped["tool_version"] == "20250905_00"
        assert dumped["connected_account_id"] == "ca_xxx"
        assert dumped["arguments"] == {"recipient_email": "a@b.c"}
        assert "input_props" in dumped

    def test_description_lists_parameters_and_configured_arguments(self):
        node = _build_node(description="Send an email.", arguments={"body": "Configured body"})

        assert node.description.startswith("Send an email.")
        assert "Required Parameters:" in node.description
        assert "- recipient_email" in node.description
        assert "Optional Parameters:" in node.description
        assert "- subject" in node.description
        assert "Already configured parameters, that can be overridden:" in node.description
        assert "- body: Configured body" in node.description

    def test_description_falls_back_to_the_slugs_when_none_is_supplied(self):
        # Without a description the agent would otherwise see only the node name, which says
        # nothing about which action this is.
        node = _build_node()

        assert node.description.startswith("Executes the Composio tool GMAIL_SEND_EMAIL from the gmail toolkit.")

    def test_to_dict_excludes_input_schema(self):
        node = _build_node()

        assert "input_schema" not in node.to_dict()

    def test_to_dict_persists_the_supplied_description_without_the_parameter_listing(self):
        node = _build_node(description="Send an email.")

        assert "Required Parameters:" in node.description
        assert node.to_dict()["description"] == "Send an email."

    def test_description_does_not_compound_across_save_and_load_cycles(self):
        # The parameter listing is appended onto whatever `description` holds, so persisting the
        # generated text would make every save/load cycle append another copy of the listing.
        node = _build_node(description="Send an email.")

        for _ in range(3):
            node = _build_node(description=node.to_dict()["description"])

        assert node.description.count("Required Parameters:") == 1
        assert node.description.count("Optional Parameters:") == 1
        assert node.description.startswith("Send an email.")


class TestComposioExecute:
    def test_execute_posts_to_the_tool_execute_endpoint(self):
        node = _build_node(arguments={"body": "Configured body"})
        client = MagicMock()
        client.request = MagicMock(return_value=_mock_response(json_payload={"data": {"id": "1"}, "successful": True}))
        node.client = client

        result = node.execute(
            node.input_schema(recipient_email="a@b.c", **{"extra-header": "x"}),
        )

        assert result == {"content": {"id": "1"}}
        call = client.request.call_args.kwargs
        assert call["method"] == "POST"
        assert call["url"] == "https://backend.composio.dev/api/v3/tools/execute/GMAIL_SEND_EMAIL"
        assert call["headers"] == {"x-api-key": "key", "Content-Type": "application/json"}
        assert call["json"] == {
            "arguments": {
                "body": "Configured body",
                "recipient_email": "a@b.c",
                "is_html": False,
                "extra-header": "x",
            },
            "user_id": "project-uuid",
        }

    def test_runtime_input_overrides_a_configured_argument(self):
        node = _build_node(arguments={"body": "Configured body"})
        client = MagicMock()
        client.request = MagicMock(return_value=_mock_response(json_payload={"data": {}, "successful": True}))
        node.client = client

        node.execute(node.input_schema(recipient_email="a@b.c", body="Runtime body"))

        assert client.request.call_args.kwargs["json"]["arguments"]["body"] == "Runtime body"

    def test_configured_argument_survives_the_schema_default(self):
        # `is_html` declares `default: false`, so a runtime input that leaves it out would otherwise
        # carry that default into the payload and silently discard the configured value.
        node = _build_node(arguments={"is_html": True})
        client = MagicMock()
        client.request = MagicMock(return_value=_mock_response(json_payload={"data": {}, "successful": True}))
        node.client = client

        node.execute(node.input_schema(recipient_email="a@b.c", body="Body"))

        assert client.request.call_args.kwargs["json"]["arguments"]["is_html"] is True

    def test_tool_version_is_sent_as_version(self):
        node = _build_node(tool_version="20250905_00")
        client = MagicMock()
        client.request = MagicMock(return_value=_mock_response(json_payload={"data": {}, "successful": True}))
        node.client = client

        node.execute(node.input_schema(recipient_email="a@b.c", body="Body"))

        assert client.request.call_args.kwargs["json"]["version"] == "20250905_00"

    def test_version_is_omitted_when_no_tool_version_is_pinned(self):
        node = _build_node()
        client = MagicMock()
        client.request = MagicMock(return_value=_mock_response(json_payload={"data": {}, "successful": True}))
        node.client = client

        node.execute(node.input_schema(recipient_email="a@b.c", body="Body"))

        assert "version" not in client.request.call_args.kwargs["json"]

    def test_connected_account_id_is_omitted_when_unset(self):
        node = _build_node()
        client = MagicMock()
        client.request = MagicMock(return_value=_mock_response(json_payload={"data": {}, "successful": True}))
        node.client = client

        node.execute(node.input_schema(recipient_email="a@b.c", body="Body"))

        assert "connected_account_id" not in client.request.call_args.kwargs["json"]

    def test_connected_account_id_is_sent_only_when_set(self):
        node = _build_node(connected_account_id="ca_1")
        client = MagicMock()
        client.request = MagicMock(return_value=_mock_response(json_payload={"data": {}, "successful": True}))
        node.client = client

        node.execute(node.input_schema(recipient_email="a@b.c", body="Body"))

        assert client.request.call_args.kwargs["json"]["connected_account_id"] == "ca_1"

    def test_output_is_raw_text_when_optimized_for_agents(self):
        node = _build_node(is_optimized_for_agents=True)
        client = MagicMock()
        client.request = MagicMock(
            return_value=_mock_response(json_payload={"data": {}, "successful": True}, text="raw body")
        )
        node.client = client

        result = node.execute(node.input_schema(recipient_email="a@b.c", body="Body"))

        assert result == {"content": "raw body"}

    def test_failed_tool_call_raises_even_though_the_status_is_200(self):
        node = _build_node()
        client = MagicMock()
        client.request = MagicMock(
            return_value=_mock_response(
                json_payload={"data": {}, "error": "Invalid recipient", "successful": False},
            )
        )
        node.client = client

        with pytest.raises(ToolExecutionException) as exc_info:
            node.execute(node.input_schema(recipient_email="a@b.c", body="Body"))

        assert "Invalid recipient" in str(exc_info.value)
        assert exc_info.value.recoverable is False
        assert client.request.call_count == 1  # the node itself does not retry an execute

    @pytest.mark.parametrize(
        ("status_code", "json_payload"),
        [(200, {"successful": False, "error": "Invalid recipient"}), (500, {}), (400, {})],
    )
    def test_run_reports_every_failure_as_recoverable(self, status_code, json_payload):
        # Characterization, not endorsement: `Node._handle_failure` derives `recoverable` from the
        # exception type, so the `recoverable=` argument at the raise site and the RECOVERABLE_CODES
        # split never reach the result - every failure is reported as recoverable and the agent may
        # re-issue the call. Update this test if `_handle_failure` starts honouring the argument.
        node = _build_node()
        client = MagicMock()
        client.closed = False
        client.is_closed = MagicMock(return_value=False)
        client.request = MagicMock(return_value=_mock_response(status_code=status_code, json_payload=json_payload))
        node.client = client

        result = node.run(input_data={"recipient_email": "a@b.c", "body": "Body"})

        assert result.status == RunnableStatus.FAILURE
        assert result.error.recoverable is True
        assert client.request.call_count == 1

    @pytest.mark.parametrize(
        ("status_code", "recoverable"),
        [(400, True), (429, True), (500, False)],
    )
    def test_error_status_raises(self, status_code, recoverable):
        node = _build_node()
        client = MagicMock()
        client.request = MagicMock(return_value=_mock_response(status_code=status_code, text="boom"))
        node.client = client

        with pytest.raises(ToolExecutionException) as exc_info:
            node.execute(node.input_schema(recipient_email="a@b.c", body="Body"))

        assert str(status_code) in str(exc_info.value)
        assert exc_info.value.recoverable is recoverable
        assert client.request.call_count == 1

    def test_run_reports_failure_when_the_tool_reports_failure(self):
        node = _build_node()
        client = MagicMock()
        client.request = MagicMock(
            return_value=_mock_response(json_payload={"data": {}, "error": "boom", "successful": False})
        )
        node.client = client

        # `run` funnels through `execute_with_retry`, which calls `ensure_client` and would replace the
        # stub above with a live `requests` client - the assertion below would then pass on a real
        # network failure instead of on the tool-level failure this test is about.
        with patch.object(Composio, "ensure_client", MagicMock()):
            result = node.run(input_data={"recipient_email": "a@b.c", "body": "Body"})

        assert result.status.value == "failure"
        assert "boom" in str(result.error.to_dict())
        assert client.request.call_count == 1


class TestComposioAsync:
    def test_composio_has_native_async(self):
        assert Composio.execute_async is not Node.execute_async

    @pytest.mark.asyncio
    async def test_execute_async_success(self):
        node = _build_node()
        client = MagicMock()
        client.request = AsyncMock(return_value=_mock_response(json_payload={"data": {"id": "1"}, "successful": True}))

        with patch.object(Composio, "get_async_client", AsyncMock(return_value=client)):
            result = await node.run_async(input_data={"recipient_email": "a@b.c", "body": "Body"})

        assert result.status.value == "success"
        assert result.output["content"] == {"id": "1"}
        client.request.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_execute_async_fails_when_the_tool_reports_failure(self):
        node = _build_node()
        client = MagicMock()
        client.request = AsyncMock(
            return_value=_mock_response(json_payload={"data": {}, "error": "boom", "successful": False})
        )

        with patch.object(Composio, "get_async_client", AsyncMock(return_value=client)):
            result = await node.run_async(input_data={"recipient_email": "a@b.c", "body": "Body"})

        assert result.status.value == "failure"
        assert client.request.await_count == 1


class TestComposioConnection:
    def test_conn_params_carry_the_api_key(self):
        connection = ComposioConnection(api_key="key")

        assert connection.conn_params == {"x-api-key": "key", "Content-Type": "application/json"}
        assert connection.type == "dynamiq.connections.Composio"

    @pytest.mark.asyncio
    async def test_connection_supports_connect_async(self):
        import httpx

        connection = ComposioConnection(api_key="key")
        client = await connection.connect_async()
        try:
            assert isinstance(client, httpx.AsyncClient)
        finally:
            await client.aclose()
