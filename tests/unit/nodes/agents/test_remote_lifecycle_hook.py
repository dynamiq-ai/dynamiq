import json
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
import yaml
from litellm import ModelResponse
from pydantic import ValidationError

from dynamiq.callbacks.base import BaseCallbackHandler
from dynamiq.connections import Exa as ExaConnection
from dynamiq.connections import Http, HTTPMethod
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.flows import Flow
from dynamiq.nodes.agents.exceptions import HookStopException
from dynamiq.nodes.agents.hooks import (
    HookContext,
    HookPoint,
    HookRunner,
    RemoteLifecycleHook,
    ToolCall,
    ToolPolicyHook,
    TransformHook,
    current_hook_ctx,
)
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.node import ErrorHandling, InputTransformer
from dynamiq.nodes.tools import RemoteContextTool
from dynamiq.nodes.tools.context_manager import ContextManagerTool
from dynamiq.nodes.tools.exa_search import ExaTool
from dynamiq.nodes.types import InferenceMode
from dynamiq.prompts import Message, MessageRole, Prompt
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.workflow import Workflow
from tests.helpers.agent_hooks import ALL_MODES, build_agent, encode, final_step, scripted, tool_step, use_mode


class CallRecorder(BaseCallbackHandler):
    def __init__(self):
        self.calls = []

    def on_node_execute_run(self, serialized, **kwargs):
        if call := kwargs.get("model_call"):
            self.calls.append(call)


def test_real_workflow_keeps_platform_run_identity_for_hooks_and_remote_child_tools():
    hook, _ = make_hook()
    hook_client = hook._client
    tool = RemoteContextTool(
        name="tenant_query",
        description="Query an authorized resource kind.",
        parameters={"type": "object", "properties": {"kind": {"type": "string"}}, "required": ["kind"]},
        connection=Http(url="https://control.example/query", method=HTTPMethod.POST),
    )
    tool.init_components()
    tool.client = MagicMock()
    tool.client.closed = False
    reply = MagicMock(status_code=200)
    reply.__enter__.return_value = reply
    reply.iter_content.return_value = [b'{"items":[]}']
    tool.client.post.return_value = reply
    agent = build_agent(hooks=[hook], tools=[tool])
    hook._client = hook_client
    config = context().config
    platform_run_id = str(uuid4())
    config.run_id = platform_run_id
    observer = MagicMock(spec=BaseCallbackHandler)
    config.callbacks = [observer]
    with scripted(agent, InferenceMode.XML, [tool_step("tenant_query", kind="alerts"), final_step("done")]) as recorder:
        result = Workflow(flow=Flow(nodes=[agent])).run({"input": "Query alerts"}, config=config)
    assert result.status == RunnableStatus.SUCCESS
    requests = [call.kwargs["json"] for call in hook_client.post.call_args_list]
    requests.append(tool.client.post.call_args.kwargs["json"])
    assert {request["run_id"] for request in requests} == {platform_run_id}
    assert all(request["app_id"] == "app-1" for request in requests)
    assert all(request["deployment_id"] == "deployment-1" for request in requests)
    child_ids = {str(call.kwargs["run_id"]) for call in observer.on_node_execute_start.call_args_list}
    assert child_ids
    assert platform_run_id not in child_ids
    assert "opaque-control-token" not in recorder.sent_text()


def test_platform_yaml_resolves_hook_managed_connection_and_round_trips(tmp_path):
    data = {
        "connections": {
            "model-connection": {"type": "dynamiq.connections.OpenAI", "api_key": "test"},
            "control-connection": {
                "type": "dynamiq.connections.Http",
                "url": "https://control.example/check",
                "method": "POST",
            },
        },
        "nodes": {
            "agent": {
                "type": "dynamiq.nodes.agents.Agent",
                "name": "agent",
                "role": "Answer the question.",
                "llm": {
                    "type": "dynamiq.nodes.llms.OpenAI",
                    "model": "gpt-4o-mini",
                    "connection": "model-connection",
                    "num_retries": 0,
                },
                "hooks": [
                    {"type": "dynamiq.nodes.agents.hooks.RemoteLifecycleHook", "connection": "control-connection"}
                ],
            },
        },
        "flows": {"main": {"nodes": ["agent"]}},
        "workflows": {"export": {"flow": "main"}},
    }
    source = tmp_path / "platform.yaml"
    source.write_text(yaml.safe_dump(data))
    workflow = Workflow.from_yaml_file(source, init_components=True)
    hook = workflow.flow.nodes[0].hooks[0]
    assert workflow.flow.nodes[0].llm.num_retries == 0
    assert isinstance(hook, RemoteLifecycleHook)
    assert hook.connection.id == "control-connection"
    assert hook.connection.url == "https://control.example/check"
    target = tmp_path / "roundtrip.yaml"
    workflow.to_yaml_file(target)
    assert (
        Workflow.from_yaml_file(target, init_components=True).flow.nodes[0].hooks[0].connection.id
        == "control-connection"
    )


def test_model_retry_override_is_forwarded_only_when_configured():
    node = OpenAI(connection=OpenAIConnection(api_key="test"), model="gpt-4o-mini")
    prompt = Prompt(messages=[])
    parameters = node._build_completion_params([], RunnableConfig(), prompt=prompt)
    assert "num_retries" not in parameters
    node.num_retries = 0
    assert node._build_completion_params([], RunnableConfig(), prompt=prompt)["num_retries"] == 0
    with pytest.raises(ValidationError):
        OpenAI(connection=OpenAIConnection(api_key="test"), model="gpt-4o-mini", num_retries=-1)


def test_tool_retry_rechecks_access_before_repeating_transport(mocker):
    hook, response = make_hook()
    ctx = context()
    ctx.hooks = [("lifecycle", hook)]
    response.iter_content.side_effect = [[b'{"allow":true}'], [b'{"allow":false}']]
    transport = mocker.patch("requests.request", side_effect=ValueError("temporary provider error"))
    tool = ExaTool(
        connection=ExaConnection(api_key="test"),
        error_handling=ErrorHandling(max_retries=1, retry_interval_seconds=0),
    )
    token = current_hook_ctx.set(ctx)
    try:
        result = tool.run_sync({"query": "public query"}, config=ctx.config)
    finally:
        current_hook_ctx.reset(token)
    assert result.status == RunnableStatus.FAILURE
    assert transport.call_count == 1
    assert hook._client.post.call_count == 2
    assert all(call.kwargs["json"]["phase"] == "before_tool" for call in hook._client.post.call_args_list)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_provider_recovery_rechecks_access_before_repeating_transport(mocker, asynchronous):
    hook, response = make_hook()
    ctx = context()
    ctx.hooks = [("lifecycle", hook)]
    recorder = CallRecorder()
    ctx.config.callbacks = [recorder]
    response.iter_content.side_effect = [
        [b'{"allow":true}'],
        [b'{"allow":false}'],
    ]
    llm = OpenAI(
        connection=OpenAIConnection(api_key="test"),
        model="gpt-4o-mini",
        prompt=Prompt(messages=[Message(role=MessageRole.USER, content="private-prompt")]),
        error_handling=ErrorHandling(max_retries=0),
    )
    mocker.patch.object(llm, "_recover_completion_params", return_value={})
    transport = (
        AsyncMock(side_effect=ValueError("unsupported parameter"))
        if asynchronous
        else MagicMock(side_effect=ValueError("unsupported parameter"))
    )
    mocker.patch.object(llm, "_acall_completion" if asynchronous else "_call_completion", transport)
    token = current_hook_ctx.set(ctx)
    try:
        result = await llm.run_async({}, config=ctx.config) if asynchronous else llm.run_sync({}, config=ctx.config)
    finally:
        current_hook_ctx.reset(token)
    assert result.status == RunnableStatus.FAILURE
    assert transport.call_count == 1
    assert hook._client.post.call_count == 2
    assert [call["status"] for call in recorder.calls] == ["started", "failed"]
    assert len({call["id"] for call in recorder.calls}) == 1
    assert "private-prompt" not in str(recorder.calls)


def make_hook(response=None):
    hook = RemoteLifecycleHook(connection=Http(url="https://control.example/check", method=HTTPMethod.POST))
    hook.init_components()
    reply = MagicMock()
    reply.status_code = 200
    reply.iter_content.return_value = [json.dumps(response or {"allow": True}).encode()]
    reply.__enter__.return_value = reply
    hook._client = MagicMock()
    hook._client.post.return_value = reply
    return hook, reply


def context():
    config = RunnableConfig(
        run_id="run-1",
        trusted_context={
            "app_id": "app-1",
            "deployment_id": "deployment-1",
            "control_context": {"token": "opaque-control-token"},
        },
    )
    return HookContext(config=config, trusted=config.trusted_context)


@pytest.mark.parametrize("point", [HookPoint.BEFORE_MODEL, HookPoint.BEFORE_TOOL])
def test_remote_guard_sends_only_control_context_and_platform_identity(point):
    hook, _ = make_hook()
    ctx = context()
    call = ToolCall(name="private-tool", input={"secret": "private-tool-arguments"})
    value = [{"content": "private-prompt"}] if point == HookPoint.BEFORE_MODEL else call.input
    result = HookRunner([hook]).run(point, ctx, value, call=call)
    assert result.value == value
    request = hook._client.post.call_args.kwargs
    assert request["json"] == {
        "phase": point.value,
        "run_id": "run-1",
        "app_id": "app-1",
        "deployment_id": "deployment-1",
        "control_context": {"token": "opaque-control-token"},
    }
    assert request["allow_redirects"] is False
    assert request["proxies"] == {"http": "", "https": "", "all": ""}
    assert "private-prompt" not in json.dumps(request["json"])
    assert "private-tool-arguments" not in json.dumps(request["json"])


@pytest.mark.parametrize("reply", [{"allow": False}, {"allow": "true"}, {}, {"reason": "private-response"}])
def test_remote_denial_or_invalid_reply_stops_without_exposing_response(reply):
    hook, response = make_hook()
    response.iter_content.return_value = [json.dumps(reply).encode()]
    with pytest.raises(HookStopException) as error:
        HookRunner([hook]).run(HookPoint.BEFORE_MODEL, context(), [])
    assert "private-response" not in str(error.value)
    assert "opaque-control-token" not in str(error.value)


@pytest.mark.parametrize("status", [301, 403, 500])
def test_remote_http_failure_is_never_followed_or_allowed(status):
    hook, response = make_hook()
    response.status_code = status
    with pytest.raises(HookStopException):
        HookRunner([hook]).run(HookPoint.BEFORE_MODEL, context(), [])


def test_remote_failure_and_large_response_stop_with_sanitized_error():
    hook, response = make_hook()
    hook._client.post.side_effect = RuntimeError("secret-token and private URL")
    with pytest.raises(HookStopException) as error:
        HookRunner([hook]).run(HookPoint.BEFORE_MODEL, context(), [])
    assert "secret-token" not in str(error.value)
    hook._client.post.side_effect = None
    response.iter_content.return_value = [b"x" * 4097]
    with pytest.raises(HookStopException):
        HookRunner([hook]).run(HookPoint.BEFORE_MODEL, context(), [])


@pytest.mark.parametrize(
    "url",
    ["http://control.example/check", "https://user:secret@control.example/check", "https://control.example/?token=x"],
)
def test_guard_rejects_unsafe_fixed_endpoint_configuration(url):
    with pytest.raises(ValidationError):
        RemoteLifecycleHook(connection=Http(url=url, method=HTTPMethod.POST))


def test_guard_cannot_fail_open_and_control_context_is_not_checkpointed():
    with pytest.raises(ValidationError):
        RemoteLifecycleHook(
            connection=Http(url="https://control.example/check", method=HTTPMethod.POST), on_error="skip"
        )
    assert "opaque-control-token" not in json.dumps(context().config.to_checkpoint_dict())


def provider_response(mode, step, index=0):
    return ModelResponse(
        model="gpt-4o",
        choices=[{"message": {"role": "assistant", **encode(mode, step, index)}, "finish_reason": "stop"}],
        usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    )


@pytest.mark.parametrize("mode", ALL_MODES)
def test_agent_rechecks_each_model_and_tool_call_without_exposing_context(mode, mocker):
    hook, _ = make_hook()
    client = hook._client
    agent = build_agent(hooks=[hook])
    hook._client = client
    use_mode(agent, mode)
    provider = mocker.patch.object(
        agent.llm,
        "_call_completion",
        side_effect=[
            provider_response(mode, tool_step("search", query="public question")),
            provider_response(mode, final_step("done"), 1),
        ],
    )
    result = agent.run_sync(
        {"input": "hello", "context": {"control_context": {"token": "model-controlled-forgery"}}},
        config=context().config,
    )
    assert result.status == RunnableStatus.SUCCESS
    assert provider.call_count == 2
    assert [call.kwargs["json"]["phase"] for call in client.post.call_args_list] == [
        "before_model",
        "before_tool",
        "before_model",
    ]
    assert "opaque-control-token" not in str(provider.call_args_list)
    assert all(
        call.kwargs["json"]["control_context"] == {"token": "opaque-control-token"}
        for call in client.post.call_args_list
    )
    assert "opaque-control-token" not in json.dumps(agent.to_dict(for_tracing=True))


def test_summary_rechecks_between_retry_calls_and_denial_prevents_next_model(mocker):
    hook, response = make_hook()
    ctx = context()
    ctx.hooks = [("lifecycle", hook)]
    llm = OpenAI(connection=OpenAIConnection(api_key="test"), model="gpt-4o-mini")
    provider = mocker.patch.object(
        llm,
        "_call_completion",
        return_value=ModelResponse(choices=[{"message": {"role": "assistant", "content": ""}}]),
    )
    tool = ContextManagerTool(llm=llm)
    response.iter_content.side_effect = [[b'{"allow":true}'], [b'{"allow":false}']]
    token = current_hook_ctx.set(ctx)
    try:
        with pytest.raises(HookStopException):
            tool._call_llm_for_summary([Message(role=MessageRole.USER, content="private history")], config=ctx.config)
    finally:
        current_hook_ctx.reset(token)
    assert provider.call_count == 1
    assert hook._client.post.call_count == 2


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_node_model_retry_authorizes_each_attempt_once(mocker, asynchronous):
    hook, _ = make_hook()
    ctx = context()
    ctx.hooks = [("lifecycle", hook)]
    llm = OpenAI(
        connection=OpenAIConnection(api_key="test"),
        model="gpt-4o-mini",
        prompt=Prompt(messages=[Message(role=MessageRole.USER, content="hello")]),
        error_handling=ErrorHandling(max_retries=1, retry_interval_seconds=0),
    )
    response = provider_response(InferenceMode.XML, final_step("done"))
    provider = (
        AsyncMock(side_effect=[RuntimeError("synthetic transient"), response])
        if asynchronous
        else MagicMock(side_effect=[RuntimeError("synthetic transient"), response])
    )
    mocker.patch.object(llm, "_acall_completion" if asynchronous else "_call_completion", provider)
    token = current_hook_ctx.set(ctx)
    try:
        result = await llm.run_async({}, config=ctx.config) if asynchronous else llm.run_sync({}, config=ctx.config)
    finally:
        current_hook_ctx.reset(token)
    assert result.status == RunnableStatus.SUCCESS
    assert provider.call_count == 2
    assert hook._client.post.call_count == 2


def test_agent_denial_at_model_dispatch_is_not_retried(mocker):
    hook, _ = make_hook({"allow": False})
    client = hook._client
    agent = build_agent(hooks=[hook], tools=[])
    hook._client = client
    agent.error_handling = ErrorHandling(max_retries=2, retry_interval_seconds=0)
    agent.llm.error_handling = ErrorHandling(max_retries=2, retry_interval_seconds=0)
    provider = mocker.patch.object(agent.llm, "_call_completion")
    result = agent.run_sync({"input": "hello"}, config=context().config)
    assert result.status == RunnableStatus.FAILURE
    assert issubclass(result.error.type, HookStopException)
    provider.assert_not_called()
    assert client.post.call_count == 1


def test_summary_authorizes_once_per_model_dispatch(mocker):
    hook, _ = make_hook()
    ctx = context()
    ctx.hooks = [("lifecycle", hook)]
    llm = OpenAI(connection=OpenAIConnection(api_key="test"), model="gpt-4o-mini")
    provider = mocker.patch.object(
        llm,
        "_call_completion",
        return_value=ModelResponse(
            choices=[{"message": {"role": "assistant", "content": "summary"}}],
        ),
    )
    tool = ContextManagerTool(llm=llm)
    token = current_hook_ctx.set(ctx)
    try:
        assert (
            tool._call_llm_for_summary([Message(role=MessageRole.USER, content="history")], config=ctx.config)
            == "summary"
        )
    finally:
        current_hook_ctx.reset(token)
    assert provider.call_count == hook._client.post.call_count == 1


@pytest.mark.parametrize("allow_search", [True, False])
def test_other_hooks_keep_transforms_and_tool_policy_at_logical_boundary(mocker, allow_search):
    hook, _ = make_hook()
    client = hook._client
    agent = build_agent(
        hooks=[
            hook,
            TransformHook(tools=["search"], input_transformer=InputTransformer(selector={"query": "trusted rewrite"})),
            ToolPolicyHook(tools=["search"], allow_if={"search_allowed": True}),
        ]
    )
    hook._client = client
    use_mode(agent, InferenceMode.FUNCTION_CALLING)
    search = next(tool for tool in agent.tools if tool.name == "search")
    transport = mocker.spy(type(search), "execute")
    provider = mocker.patch.object(
        agent.llm,
        "_call_completion",
        side_effect=[
            provider_response(InferenceMode.FUNCTION_CALLING, tool_step("search", query="model query")),
            provider_response(InferenceMode.FUNCTION_CALLING, final_step("done"), 1),
        ],
    )
    config = context().config
    config.trusted_context["search_allowed"] = allow_search
    observer = MagicMock(spec=BaseCallbackHandler)
    config.callbacks = [observer]
    result = agent.run_sync({"input": "hello"}, config=config)
    assert result.status == RunnableStatus.SUCCESS
    assert provider.call_count == 2
    assert transport.call_count == int(allow_search)
    if allow_search:
        assert transport.call_args.kwargs["input_data"].query == "trusted rewrite"
    assert client.post.call_count == 2 + int(allow_search)
    assert observer.on_node_execute_start.called
    assert observer.on_node_execute_end.called


@pytest.mark.parametrize("allow_query", [True, False])
def test_remote_context_tool_authorizes_once_before_private_request(mocker, allow_query):
    hook, response = make_hook()
    client = hook._client
    if not allow_query:
        response.iter_content.side_effect = [[b'{"allow":true}'], [b'{"allow":false}']]
    tool = RemoteContextTool(
        name="tenant-query",
        description="Query authorized records.",
        parameters={"type": "object", "properties": {"kind": {"type": "string"}}, "required": ["kind"]},
        connection=Http(url="https://control.example/query", method=HTTPMethod.POST),
    )
    agent = build_agent(hooks=[hook], tools=[tool])
    hook._client = client
    query = MagicMock()
    query.closed = False
    reply = MagicMock(status_code=200)
    reply.__enter__.return_value = reply
    reply.iter_content.return_value = [b'{"items":[]}']
    query.post.return_value = reply
    tool.client = query
    use_mode(agent, InferenceMode.FUNCTION_CALLING)
    provider = mocker.patch.object(
        agent.llm,
        "_call_completion",
        side_effect=[
            provider_response(InferenceMode.FUNCTION_CALLING, tool_step("tenant-query", kind="alerts")),
            provider_response(InferenceMode.FUNCTION_CALLING, final_step("done"), 1),
        ],
    )
    result = agent.run_sync({"input": "hello"}, config=context().config)
    assert result.status == (RunnableStatus.SUCCESS if allow_query else RunnableStatus.FAILURE)
    assert provider.call_count == 1 + int(allow_query)
    phases = ["before_model", "before_tool"] + (["before_model"] if allow_query else [])
    assert [call.kwargs["json"]["phase"] for call in client.post.call_args_list] == phases
    assert query.post.call_count == int(allow_query)
    if allow_query:
        assert query.post.call_args.kwargs["json"]["control_context"] == {"token": "opaque-control-token"}
    assert "opaque-control-token" not in str(provider.call_args_list)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_parameter_recovery_authorizes_each_provider_attempt_once(mocker, asynchronous):
    hook, _ = make_hook()
    ctx = context()
    ctx.hooks = [("lifecycle", hook)]
    llm = OpenAI(
        connection=OpenAIConnection(api_key="test"),
        model="gpt-4o-mini",
        prompt=Prompt(messages=[Message(role=MessageRole.USER, content="hello")]),
    )
    mocker.patch.object(llm, "_recover_completion_params", return_value={})
    response = provider_response(InferenceMode.XML, final_step("done"))
    provider = (
        AsyncMock(side_effect=[ValueError("unsupported parameter"), response])
        if asynchronous
        else MagicMock(side_effect=[ValueError("unsupported parameter"), response])
    )
    mocker.patch.object(llm, "_acall_completion" if asynchronous else "_call_completion", provider)
    token = current_hook_ctx.set(ctx)
    try:
        result = await llm.run_async({}, config=ctx.config) if asynchronous else llm.run_sync({}, config=ctx.config)
    finally:
        current_hook_ctx.reset(token)
    assert result.status == RunnableStatus.SUCCESS
    assert provider.call_count == hook._client.post.call_count == 2


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_tool_retry_authorizes_each_dispatch_once(mocker, asynchronous):
    hook, _ = make_hook()
    ctx = context()
    ctx.hooks = [("lifecycle", hook)]
    tool = build_agent().tools[0]
    tool.error_handling = ErrorHandling(max_retries=1, retry_interval_seconds=0)
    provider = mocker.patch.object(
        type(tool),
        "execute",
        side_effect=[
            RuntimeError("synthetic transient"),
            {"content": "public result"},
        ],
    )
    token = current_hook_ctx.set(ctx)
    try:
        result = (
            await tool.run_async({"query": "hello"}, config=ctx.config)
            if asynchronous
            else tool.run_sync(
                {"query": "hello"},
                config=ctx.config,
            )
        )
    finally:
        current_hook_ctx.reset(token)
    assert result.status == RunnableStatus.SUCCESS
    assert provider.call_count == hook._client.post.call_count == 2
