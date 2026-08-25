import asyncio
import time

import pytest

from dynamiq.callbacks.tracing import TracingCallbackHandler
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import Node, NodeGroup
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types.cancellation import CancellationConfig, CancellationToken
from dynamiq.types.feedback import ApprovalConfig, ApprovalInputData
from dynamiq.types.mocking import DEFAULT_MOCK_MARKER, MockConfig, MockPolicy, RunMockConfig

LATENCY = 0.3
TIMING_TOLERANCE = 0.2


class RecordingToolNode(Node):
    """Tool node that records every real execution, so mocks are provable by absence."""

    group: NodeGroup = NodeGroup.TOOLS
    name: str = "recording-tool"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._calls: list = []

    @property
    def calls(self) -> list:
        return self._calls

    def execute(self, input_data: dict, config: RunnableConfig = None, **kwargs) -> dict:
        self._calls.append(input_data)
        return {"content": "real result"}


class AsyncRecordingToolNode(RecordingToolNode):
    """Same, but with a native async execute so the async seam is exercised."""

    name: str = "async-recording-tool"

    async def execute_async(self, input_data: dict, config: RunnableConfig = None, **kwargs) -> dict:
        self._calls.append(input_data)
        return {"content": "real async result"}


class WriterNode(RecordingToolNode):
    """A non-tool node, for group-scoping assertions."""

    group: NodeGroup = NodeGroup.WRITERS
    name: str = "writer"


class TestMockConfigRender:
    """The synthetic result a mocked node produces."""

    def test_echo_is_used_when_no_response_configured(self):
        output = MockConfig(enabled=True).render(node_name="charge_card", node_id="n1", input_data={"amount": 10})

        assert output["content"].startswith(DEFAULT_MOCK_MARKER)
        assert "charge_card" in output["content"]
        assert "amount" in output["content"]
        assert output["is_mocked"] is True

    def test_string_output_is_rendered_against_the_input(self):
        mock = MockConfig(enabled=True, output="refunded {{ input.amount }} to {{ input.user }}")

        output = mock.render(node_name="n", node_id="n1", input_data={"amount": 25, "user": "ada"})

        assert output["content"] == f"{DEFAULT_MOCK_MARKER} refunded 25 to ada"

    def test_dict_output_is_returned_verbatim(self):
        """An authored dict owns the whole shape — nothing is added, annotated or renamed."""
        authored = {"content": "ok", "status_code": 201, "receipt": "r-9"}

        output = MockConfig(enabled=True, output=authored).render(node_name="n", node_id="n1", input_data={})

        assert output == authored

    def test_a_dict_output_for_a_non_tool_shape_is_left_alone(self):
        """The docstring tells authors to match the real shape; adding `content` would break that."""
        authored = {"documents": [{"text": "hit"}], "score": 0.9}

        output = MockConfig(enabled=True, output=authored).render(node_name="n", node_id="n1", input_data={})

        assert output == authored
        assert "content" not in output
        assert "is_mocked" not in output

    def test_non_string_content_is_left_untouched_by_annotation(self):
        output = MockConfig(enabled=True, output={"content": {"rows": 3}}).render(
            node_name="n", node_id="n1", input_data={}
        )

        assert output["content"] == {"rows": 3}

    def test_annotation_can_be_disabled(self):
        output = MockConfig(enabled=True, output="clean", marker=None).render(
            node_name="n", node_id="n1", input_data={}
        )

        assert output["content"] == "clean"

    def test_marker_is_not_applied_twice(self):
        mock = MockConfig(enabled=True, output=f"{DEFAULT_MOCK_MARKER} already marked")

        output = mock.render(node_name="n", node_id="n1", input_data={})

        assert output["content"].count(DEFAULT_MOCK_MARKER) == 1

    def test_custom_marker_is_honoured(self):
        output = MockConfig(enabled=True, output="hi", marker="[SIMULATED]").render(
            node_name="n", node_id="n1", input_data={}
        )

        assert output["content"] == "[SIMULATED] hi"

    def test_render_does_not_mutate_the_configured_output(self):
        mock = MockConfig(enabled=True, output={"content": "ok"})

        first = mock.render(node_name="n", node_id="n1", input_data={})
        second = mock.render(node_name="n", node_id="n1", input_data={})

        assert mock.output == {"content": "ok"}
        assert first == second

    def test_echoed_input_is_bounded(self):
        """The echo lands in the agent prompt and the trace, so it must stay bounded."""
        output = MockConfig(enabled=True).render(node_name="n", node_id="n1", input_data={"blob": "x" * 50_000})

        assert len(output["content"]) < 2_000
        assert "x" * 50_000 not in output["content"]

    def test_a_wide_input_is_bounded_too(self):
        """Bounded by breadth as well as length — a thousand small keys must not all render."""
        output = MockConfig(enabled=True).render(
            node_name="n", node_id="n1", input_data={f"k{i}": i for i in range(1_000)}
        )

        assert len(output["content"]) < 2_000

    def test_authored_output_is_not_truncated(self):
        mock = MockConfig(enabled=True, output="y" * 5_000)

        output = mock.render(node_name="n", node_id="n1", input_data={})

        assert len(output["content"]) > 5_000

    def test_locked_without_enabled_is_inert(self):
        assert MockConfig(locked=True).is_pinned is False
        assert MockConfig(enabled=True, locked=True).is_pinned is True

    def test_negative_latency_is_rejected(self):
        with pytest.raises(ValueError):
            MockConfig(enabled=True, latency_seconds=-1)

    def test_the_default_description_renders_no_template(self):
        """The zero-config path must not run Jinja — a brace in the input is data, not code."""
        output = MockConfig(enabled=True).render(node_name="n", node_id="n1", input_data={"q": "{{ 7 * 6 }}"})

        assert "{{ 7 * 6 }}" in output["content"]
        assert "42" not in output["content"]

    def test_the_default_description_respects_a_disabled_marker(self):
        output = MockConfig(enabled=True, marker=None).render(node_name="n", node_id="n1", input_data={})

        assert not output["content"].startswith(DEFAULT_MOCK_MARKER)
        assert "was skipped" in output["content"]


class TestMockPolicyResolution:
    """Which nodes a given run mocks."""

    @staticmethod
    def resolve(run_mock: RunMockConfig, node_mock: MockConfig, group: str = "tools", name: str = "tool-a"):
        return run_mock.resolve(node_mock, {"id-a"}, {name} if name else set(), group)

    def test_node_policy_honours_node_config(self):
        assert self.resolve(RunMockConfig(), MockConfig(enabled=True)) is not None
        assert self.resolve(RunMockConfig(), MockConfig(enabled=False)) is None

    def test_all_policy_sweeps_in_configured_groups(self):
        run_mock = RunMockConfig(policy=MockPolicy.ALL)

        assert self.resolve(run_mock, MockConfig(), group="tools") is run_mock.default
        assert self.resolve(run_mock, MockConfig(), group="llms") is None
        assert self.resolve(run_mock, MockConfig(), group="agents") is None

    def test_an_unknown_group_is_rejected_rather_than_silently_matching_nothing(self):
        with pytest.raises(ValueError, match="Unknown node group"):
            RunMockConfig(policy=MockPolicy.ALL, groups={"tool"})

    def test_every_real_node_group_is_accepted(self):
        assert RunMockConfig(groups={group.value for group in NodeGroup}).groups

    def test_all_policy_group_set_is_configurable(self):
        run_mock = RunMockConfig(policy=MockPolicy.ALL, groups={"tools", "writers"})

        assert self.resolve(run_mock, MockConfig(), group="writers") is run_mock.default

    def test_all_policy_preserves_a_curated_node_output(self):
        node_mock = MockConfig(enabled=True, output="curated")

        assert self.resolve(RunMockConfig(policy=MockPolicy.ALL), node_mock) is node_mock

    def test_all_policy_uses_an_authored_output_that_was_left_switched_off(self):
        node_mock = MockConfig(enabled=False, output="curated")

        resolved = self.resolve(RunMockConfig(policy=MockPolicy.ALL), node_mock)

        assert resolved is not None
        assert resolved.output == "curated", "the generic echo must not replace an authored output"
        assert resolved.enabled is True
        assert node_mock.enabled is False, "resolution must not mutate the node's own config"

    def test_all_policy_uses_an_authored_error_that_was_left_switched_off(self):
        resolved = self.resolve(RunMockConfig(policy=MockPolicy.ALL), MockConfig(enabled=False, error="boom"))

        assert resolved.error == "boom"

    def test_all_policy_falls_back_to_the_default_for_an_empty_config(self):
        run_mock = RunMockConfig(policy=MockPolicy.ALL)

        assert self.resolve(run_mock, MockConfig()) is run_mock.default

    def test_none_policy_unmocks_ordinary_nodes(self):
        assert self.resolve(RunMockConfig(policy=MockPolicy.NONE), MockConfig(enabled=True)) is None

    def test_none_policy_cannot_unmock_a_locked_node(self):
        node_mock = MockConfig(enabled=True, locked=True)

        assert self.resolve(RunMockConfig(policy=MockPolicy.NONE), node_mock) is node_mock

    @pytest.mark.parametrize("exclude", [{"exclude_names": {"tool-a"}}, {"exclude_ids": {"id-a"}}])
    def test_exclude_matches_by_name_or_by_id(self, exclude):
        run_mock = RunMockConfig(policy=MockPolicy.ALL, **exclude)

        assert self.resolve(run_mock, MockConfig()) is None

    def test_an_id_in_exclude_names_does_not_match(self):
        """The fields are separate so a name can never silently un-mock an unrelated node."""
        run_mock = RunMockConfig(policy=MockPolicy.ALL, exclude_names={"id-a"})

        assert self.resolve(run_mock, MockConfig()) is not None

    def test_exclude_beats_node_level_config(self):
        run_mock = RunMockConfig(exclude_names={"tool-a"})

        assert self.resolve(run_mock, MockConfig(enabled=True)) is None

    def test_exclude_does_not_beat_a_locked_node(self):
        node_mock = MockConfig(enabled=True, locked=True)

        assert self.resolve(RunMockConfig(policy=MockPolicy.ALL, exclude_names={"tool-a"}), node_mock) is node_mock

    def test_unnamed_node_is_not_matched_by_exclude(self):
        run_mock = RunMockConfig(policy=MockPolicy.ALL, exclude_names={"tool-a"})

        assert run_mock.resolve(MockConfig(), {"id-b"}, set(), "tools") is run_mock.default


class TestNodeMockExecution:
    """The node never executes, but behaves like it did."""

    def test_mocked_node_is_not_executed(self):
        node = RecordingToolNode(mock=MockConfig(enabled=True))

        result = node.run(input_data={"a": 1})

        assert result.status == RunnableStatus.SUCCESS
        assert node.calls == []
        assert result.output["is_mocked"] is True

    def test_unmocked_node_is_unaffected(self):
        node = RecordingToolNode()

        result = node.run(input_data={"a": 1})

        assert result.output == {"content": "real result"}
        assert node.calls == [{"a": 1}]

    def test_run_level_all_policy_mocks_a_node_with_no_config(self):
        node = RecordingToolNode()

        result = node.run(input_data={"a": 1}, config=RunnableConfig(mock=RunMockConfig(policy=MockPolicy.ALL)))

        assert node.calls == []
        assert result.output["is_mocked"] is True

    def test_run_level_all_policy_skips_nodes_outside_the_group_set(self):
        node = WriterNode()

        node.run(input_data={"a": 1}, config=RunnableConfig(mock=RunMockConfig(policy=MockPolicy.ALL)))

        assert node.calls == [{"a": 1}]

    def test_run_level_none_policy_restores_real_execution(self):
        node = RecordingToolNode(mock=MockConfig(enabled=True))

        node.run(input_data={"a": 1}, config=RunnableConfig(mock=RunMockConfig(policy=MockPolicy.NONE)))

        assert node.calls == [{"a": 1}]

    def test_locked_node_never_executes_even_under_none_policy(self):
        node = RecordingToolNode(mock=MockConfig(enabled=True, locked=True))

        result = node.run(input_data={"a": 1}, config=RunnableConfig(mock=RunMockConfig(policy=MockPolicy.NONE)))

        assert node.calls == []
        assert result.status == RunnableStatus.SUCCESS

    def test_error_injection_fails_recoverably_without_executing(self):
        node = RecordingToolNode(mock=MockConfig(enabled=True, error="upstream 503"))

        result = node.run(input_data={"a": 1})

        assert result.status == RunnableStatus.FAILURE
        assert result.error.type is ToolExecutionException
        assert result.error.recoverable is True
        assert "upstream 503" in result.error.message
        assert node.calls == []

    def test_injected_error_message_respects_annotate(self):
        node = RecordingToolNode(mock=MockConfig(enabled=True, error="503 Service Unavailable", marker=None))

        result = node.run(input_data={})

        assert result.error.message == "503 Service Unavailable"

    def test_error_takes_precedence_over_output(self):
        node = RecordingToolNode(mock=MockConfig(enabled=True, output="ignored", error="boom"))

        assert node.run(input_data={}).status == RunnableStatus.FAILURE

    def test_input_schema_validation_still_runs_under_mock(self):
        """A dry run must still catch a malformed tool call."""

        class StrictNode(RecordingToolNode):
            def transform_input(self, *args, **kwargs):
                raise ValueError("bad input")

        node = StrictNode(mock=MockConfig(enabled=True))

        assert node.run(input_data={"a": 1}).status == RunnableStatus.FAILURE

    def test_output_transformer_applies_to_mocked_output(self):
        node = RecordingToolNode(mock=MockConfig(enabled=True, output={"content": "x", "keep": "me"}))
        node.output_transformer.path = "$.keep"

        assert node.run(input_data={}).output == "me"

    def test_mocked_run_bypasses_the_cache_layer(self):
        node = RecordingToolNode(mock=MockConfig(enabled=True))
        node.caching.enabled = True

        result = node.run(input_data={"a": 1})

        assert result.output["is_mocked"] is True
        assert node.calls == []

    def test_latency_is_simulated(self):
        node = RecordingToolNode(mock=MockConfig(enabled=True, latency_seconds=LATENCY))

        start = time.monotonic()
        node.run(input_data={})

        assert time.monotonic() - start >= LATENCY

    def test_latency_wait_is_cancellable(self):
        node = RecordingToolNode(mock=MockConfig(enabled=True, latency_seconds=30))
        token = CancellationToken()
        token.cancel()
        config = RunnableConfig(cancellation=CancellationConfig(token=token))

        start = time.monotonic()
        result = node.run(input_data={}, config=config)

        assert result.status == RunnableStatus.CANCELED
        assert time.monotonic() - start < 30 - TIMING_TOLERANCE


class TestNodeMockExecutionAsync:
    """Both async seams — the native one and the thread-offload fallback."""

    @pytest.mark.asyncio
    async def test_native_async_node_is_mocked(self):
        node = AsyncRecordingToolNode(mock=MockConfig(enabled=True))

        result = await node.run_async(input_data={"a": 1})

        assert node.calls == []
        assert result.output["is_mocked"] is True

    @pytest.mark.asyncio
    async def test_native_async_node_runs_for_real_when_unmocked(self):
        node = AsyncRecordingToolNode()

        result = await node.run_async(input_data={"a": 1})

        assert result.output == {"content": "real async result"}

    @pytest.mark.asyncio
    async def test_sync_only_node_is_mocked_on_the_async_path(self):
        node = RecordingToolNode(mock=MockConfig(enabled=True))

        result = await node.run_async(input_data={"a": 1})

        assert node.calls == []
        assert result.output["is_mocked"] is True

    @pytest.mark.asyncio
    async def test_async_latency_does_not_block_the_event_loop(self):
        node = AsyncRecordingToolNode(mock=MockConfig(enabled=True, latency_seconds=LATENCY))

        start = time.monotonic()
        await asyncio.gather(*(node.run_async(input_data={}) for _ in range(3)))
        elapsed = time.monotonic() - start

        assert elapsed < LATENCY * 3

    @pytest.mark.asyncio
    async def test_async_latency_wait_is_cancellable(self):
        node = AsyncRecordingToolNode(mock=MockConfig(enabled=True, latency_seconds=30))
        token = CancellationToken()
        token.cancel()

        start = time.monotonic()
        result = await node.run_async(
            input_data={}, config=RunnableConfig(cancellation=CancellationConfig(token=token))
        )

        assert result.status == RunnableStatus.CANCELED
        assert time.monotonic() - start < 30 - TIMING_TOLERANCE

    @pytest.mark.asyncio
    async def test_async_error_injection(self):
        node = AsyncRecordingToolNode(mock=MockConfig(enabled=True, error="nope"))

        result = await node.run_async(input_data={})

        assert result.status == RunnableStatus.FAILURE
        assert node.calls == []


class TestMockAndApproval:
    """Approval authorizes a side effect; a mocked node has none, so it must not prompt."""

    @staticmethod
    def spy_on_approval(monkeypatch) -> list[str]:
        asked: list[str] = []

        def fake_prompt(self, template, config=None):
            asked.append(template)
            return ApprovalInputData(feedback="")

        monkeypatch.setattr(Node, "send_console_approval_message", fake_prompt)
        return asked

    def test_a_mocked_node_does_not_ask_for_approval(self, monkeypatch):
        asked = self.spy_on_approval(monkeypatch)
        node = RecordingToolNode(
            approval=ApprovalConfig(enabled=True),
            mock=MockConfig(enabled=True, output="pretend done"),
        )

        result = node.run(input_data={})

        assert asked == [], "a node that cannot execute must not ask a human to authorize it"
        assert result.output["content"] == "[MOCKED] pretend done"
        assert node.calls == []

    def test_an_unmocked_node_still_asks_for_approval(self):
        """The guard must be narrow — approval is untouched when the node really runs."""
        asked: list[str] = []

        class ApprovalSpyNode(RecordingToolNode):
            def send_console_approval_message(self, template, config=None):
                asked.append(template)
                return ApprovalInputData(feedback="")

        node = ApprovalSpyNode(approval=ApprovalConfig(enabled=True))
        node.run(input_data={})

        assert len(asked) == 1
        assert node.calls == [{}]

    def test_a_rejected_approval_still_skips_an_unmocked_node(self):
        class RejectingNode(RecordingToolNode):
            def send_console_approval_message(self, template, config=None):
                return ApprovalInputData(feedback="no, do not do this")

        node = RejectingNode(approval=ApprovalConfig(enabled=True))

        result = node.run(input_data={})

        assert result.status == RunnableStatus.SKIP
        assert node.calls == []

    def test_run_level_policy_also_suppresses_the_prompt(self, monkeypatch):
        """An unattended MockPolicy.ALL run must not stall on approval-gated tools."""
        asked = self.spy_on_approval(monkeypatch)
        node = RecordingToolNode(approval=ApprovalConfig(enabled=True))

        node.run(input_data={}, config=RunnableConfig(mock=RunMockConfig(policy=MockPolicy.ALL)))

        assert asked == []
        assert node.calls == []


class TestMockObservability:
    """A mocked run must never be mistaken for a real one."""

    @staticmethod
    def node_run_metadata(config: RunnableConfig) -> dict:
        """Metadata of the single node run recorded on this config's tracing handler."""
        handler = next(cb for cb in config.callbacks if isinstance(cb, TracingCallbackHandler))
        (run,) = handler.runs.values()
        return run.metadata

    def test_trace_metadata_flags_a_mocked_run(self):
        node = RecordingToolNode(mock=MockConfig(enabled=True))
        config = RunnableConfig(callbacks=[TracingCallbackHandler()])

        node.run(input_data={}, config=config)

        assert self.node_run_metadata(config)["is_mocked"] is True

    def test_trace_metadata_flags_a_real_run(self):
        node = RecordingToolNode()
        config = RunnableConfig(callbacks=[TracingCallbackHandler()])

        node.run(input_data={}, config=config)

        assert self.node_run_metadata(config)["is_mocked"] is False

    def test_trace_metadata_flags_an_injected_failure(self):
        node = RecordingToolNode(mock=MockConfig(enabled=True, error="simulated outage"))
        config = RunnableConfig(callbacks=[TracingCallbackHandler()])

        node.run(input_data={}, config=config)

        assert self.node_run_metadata(config)["is_mocked"] is True

    def test_trace_metadata_flags_a_real_failure_as_not_mocked(self):
        class FailingNode(RecordingToolNode):
            def execute(self, input_data, config=None, **kwargs):
                raise RuntimeError("real outage")

        config = RunnableConfig(callbacks=[TracingCallbackHandler()])
        FailingNode().run(input_data={}, config=config)

        assert self.node_run_metadata(config)["is_mocked"] is False

    def test_a_caller_cannot_forge_the_mocked_flag(self):
        """`is_mocked` is framework-reserved; a stray kwarg must not mark a real run as mocked."""
        node = RecordingToolNode()
        config = RunnableConfig(callbacks=[TracingCallbackHandler()])

        result = node.run(input_data={}, config=config, is_mocked=True)

        assert result.status == RunnableStatus.SUCCESS
        assert node.calls == [{}], "the node must still really execute"
        assert self.node_run_metadata(config)["is_mocked"] is False

    def test_a_caller_cannot_forge_the_flag_on_a_failing_run(self):
        """The forged value must be dropped before validation can fail the run."""

        class FailingTransformNode(RecordingToolNode):
            def transform_input(self, *args, **kwargs):
                raise ValueError("bad input")

        node = FailingTransformNode()
        config = RunnableConfig(callbacks=[TracingCallbackHandler()])

        result = node.run(input_data={}, config=config, is_mocked=True)

        assert result.status == RunnableStatus.FAILURE
        assert self.node_run_metadata(config)["is_mocked"] is False

    def test_real_execution_does_not_receive_the_reserved_kwarg(self):
        seen: list[dict] = []

        class KwargSpyNode(RecordingToolNode):
            def execute(self, input_data, config=None, **kwargs):
                seen.append(kwargs)
                return {"content": "ok"}

        KwargSpyNode().run(input_data={})

        assert "is_mocked" not in seen[0]

    def test_to_dict_exposes_the_mock_config(self):
        node = RecordingToolNode(mock=MockConfig(enabled=True, output="canned"))

        assert node.to_dict()["mock"]["enabled"] is True
        assert node.to_dict()["mock"]["output"] == "canned"

    def test_tracing_collapses_a_disabled_mock_config(self):
        assert RecordingToolNode().to_dict(for_tracing=True)["mock"] == {"enabled": False}

    def test_tracing_keeps_an_enabled_mock_config(self):
        node = RecordingToolNode(mock=MockConfig(enabled=True, locked=True))

        assert node.to_dict(for_tracing=True)["mock"]["locked"] is True


class TestMockSerialization:
    """Config must survive the canvas round-trip."""

    def test_mock_config_is_coerced_from_a_plain_dict(self):
        node = RecordingToolNode(mock={"enabled": True, "output": "canned", "locked": True})

        assert isinstance(node.mock, MockConfig)
        assert node.mock.locked is True

    def test_node_defaults_to_no_mock(self):
        node = RecordingToolNode()

        assert node.mock.enabled is False
        assert node.resolve_mock(RunnableConfig()) is None

    def test_round_trip_through_to_dict(self):
        node = RecordingToolNode(mock=MockConfig(enabled=True, output={"content": "x"}, latency_seconds=0.5))

        restored = RecordingToolNode(mock=node.to_dict()["mock"])

        assert restored.mock == node.mock

    def test_run_config_is_checkpoint_serializable(self):
        """The policy must survive model_dump, because resume reads it back.

        `Flow._restore_config_from_checkpoint` restores `mock` (and `dry_run`,
        `max_node_workers`) from `original_config` when the caller leaves them unset, so a
        resumed dry run stays a dry run. End-to-end coverage lives in
        tests/integration/checkpoints/test_resume_restores_run_config.py; this asserts only
        the serialization step that restore depends on.
        """
        config = RunnableConfig(mock=RunMockConfig(policy=MockPolicy.ALL, exclude_names={"search"}))

        assert config.to_checkpoint_dict()["mock"]["policy"] == "all"


class TestMockOutputIsolation:
    """One config serves every call, so an authored dict must not be shared between them."""

    def test_an_authored_dict_is_not_shared_between_calls(self):
        mock = MockConfig(enabled=True, output={"documents": [{"content": "a"}]})

        first = mock.render(node_name="retriever", node_id="n1", input_data={})
        first["documents"].append({"content": "injected"})
        second = mock.render(node_name="retriever", node_id="n1", input_data={})

        assert second["documents"] == [{"content": "a"}], "a mutation downstream rewrote the mock"
        assert mock.output == {"documents": [{"content": "a"}]}, "the config itself was mutated"


class TestMockAcrossClonedExecution:
    """Map and the agent both clone a node per iteration; id-keyed config must follow the clone."""

    def test_map_carries_an_id_exclusion_onto_each_cloned_iteration(self):
        from dynamiq.nodes.operators.operators import Map

        inner = RecordingToolNode(name="search")
        mapper = Map(node=inner)
        config = RunnableConfig(mock=RunMockConfig(policy=MockPolicy.ALL, exclude_ids={inner.id}))

        result = mapper.run(input_data={"input": [{"a": 1}, {"a": 2}]}, config=config)

        assert result.status == RunnableStatus.SUCCESS
        # The clones share the fixture's call list, so this counts real executions across iterations.
        assert len(inner.calls) == 2, "an excluded node must really run in every iteration"
        assert all(out == {"content": "real result"} for out in result.output["output"]), result.output["output"]

    def test_map_still_mocks_a_node_that_was_not_excluded(self):
        from dynamiq.nodes.operators.operators import Map

        mapper = Map(node=RecordingToolNode(name="search"))
        config = RunnableConfig(mock=RunMockConfig(policy=MockPolicy.ALL))

        result = mapper.run(input_data={"input": [{"a": 1}]}, config=config)

        assert result.output["output"][0]["is_mocked"] is True

    def test_one_original_cloned_twice_keeps_both_exclusions(self):
        """`id_map` maps one old id to a set, so a node reachable twice does not lose its clone."""
        from dynamiq.nodes.cloning import carry_mock_exclusions, regenerate_node_ids

        shared = RecordingToolNode(name="search")
        original_id = shared.id
        id_map: dict[str, set[str]] = {}
        clone_a = regenerate_node_ids(shared.clone(), id_map)
        clone_b = regenerate_node_ids(shared.clone(), id_map)

        config = carry_mock_exclusions(
            RunnableConfig(mock=RunMockConfig(policy=MockPolicy.ALL, exclude_ids={original_id})), id_map
        )

        assert clone_a.resolve_mock(config) is None
        assert clone_b.resolve_mock(config) is None


class TestMockDoesNotWidenExposure:
    """A mocked run must not reveal more than a real one."""

    def test_the_description_omits_fields_the_agent_may_not_see(self):
        """In a real run a hidden credential goes to the tool and never reaches the transcript."""
        from typing import ClassVar

        from pydantic import BaseModel, Field

        class Schema(BaseModel):
            query: str = ""
            access_token: str = Field(default="", json_schema_extra={"is_accessible_to_agent": False})

        class CrmTool(RecordingToolNode):
            name: str = "crm_update"
            input_schema: ClassVar[type[Schema]] = Schema

        node = CrmTool(mock=MockConfig(enabled=True))
        result = node.run(input_data={"query": "acme", "access_token": "ghp_SECRET_TOKEN"})

        assert "ghp_SECRET_TOKEN" not in result.output["content"]
        assert "acme" in result.output["content"], "visible arguments are still described"

    def test_agent_machinery_is_not_swept_in_by_a_whole_run_dry_run(self):
        """Mocking the context manager corrupts the agent instead of protecting the world."""
        from dynamiq.nodes.tools.context_manager import ContextManagerTool
        from dynamiq.nodes.tools.todo_tools import TodoWriteTool

        run_mock = RunMockConfig(policy=MockPolicy.ALL)
        for cls in (ContextManagerTool, TodoWriteTool):
            assert cls.is_mockable is False
            assert run_mock.resolve(MockConfig(), {"x"}, {"y"}, "tools", sweepable=cls.is_mockable) is None

    def test_machinery_can_still_be_mocked_explicitly(self):
        """Opting out of the sweep must not override an author's explicit intent."""
        run_mock = RunMockConfig(policy=MockPolicy.ALL)
        explicit = MockConfig(enabled=True, output="canned")

        assert run_mock.resolve(explicit, {"x"}, {"y"}, "tools", sweepable=False) is explicit
