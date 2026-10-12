"""Hook primitives: ordering, decisions, errors, timeouts, strict config, serialization."""

import time

import pytest
from pydantic import ValidationError

from dynamiq.nodes.agents.exceptions import HookAnswerException, HookStopException, ToolBlockedException
from dynamiq.nodes.agents.hooks import (
    ALLOW,
    Block,
    BlockAs,
    Hook,
    HookContext,
    HookErrorPolicy,
    HookPoint,
    HookRunner,
    Modify,
    Skip,
    ToolCall,
    resolve_hook,
)

CALL = ToolCall(name="search", input={"query": "q"})


class Suffix(Hook):
    """Appends its tag to the value, so the order of evaluation is visible in the result."""

    tag: str = ""

    def on_input(self, ctx, text):
        return Modify(text + self.tag)

    def on_output(self, ctx, answer):
        return Modify(answer + self.tag)


class BlockIf(Hook):
    needle: str = "SSN"
    as_: BlockAs = BlockAs.OBSERVATION

    def on_input(self, ctx, text):
        return Block("nope", self.as_) if self.needle in text else ALLOW


class Crash(Hook):
    def on_input(self, ctx, text):
        raise RuntimeError("secret payload in the message: " + text)

    def before_tool(self, ctx, call):
        raise RuntimeError("boom")


class Sleepy(Hook):
    def on_input(self, ctx, text):
        time.sleep(0.5)
        return ALLOW


def run(hooks, point, value, **kwargs):
    events: list[dict] = []
    result = HookRunner(hooks).run(point, HookContext(), value, events=events, **kwargs)
    return result, events


def test_before_points_run_in_list_order_and_after_points_reversed():
    hooks = [Suffix(tag="A"), Suffix(tag="B")]
    assert run(hooks, HookPoint.ON_INPUT, "x")[0].value == "xAB"
    assert run(hooks, HookPoint.ON_OUTPUT, "x")[0].value == "xBA"


def test_hook_by_hook_a_guard_before_a_masker_sees_the_raw_value():
    class Mask(Hook):
        def on_input(self, ctx, text):
            return Modify(text.replace("SSN", "[X]"))

    with pytest.raises(HookAnswerException):
        run([BlockIf(), Mask()], HookPoint.ON_INPUT, "my SSN")
    assert run([Mask(), BlockIf()], HookPoint.ON_INPUT, "my SSN")[0].value == "my [X]"


@pytest.mark.parametrize(
    "as_, expected",
    [
        (BlockAs.FAIL, HookStopException),
        (BlockAs.ANSWER, HookAnswerException),
        (BlockAs.OBSERVATION, HookAnswerException),
    ],
)
def test_block_outcomes(as_, expected):
    with pytest.raises(expected) as info:
        run([BlockIf(as_=as_)], HookPoint.ON_INPUT, "SSN")
    assert info.value.hook == "BlockIf" and info.value.point == "on_input"


def test_block_at_a_tool_point_is_an_observation_with_the_hook_name():
    class Deny(Hook):
        def before_tool(self, ctx, call):
            return Block("denied")

    with pytest.raises(ToolBlockedException) as info:
        run([Deny(name="no-search")], HookPoint.BEFORE_TOOL, CALL.input, call=CALL)
    assert str(info.value) == "denied" and info.value.hook == "no-search"


def test_block_ends_evaluation_and_the_event_has_the_hook_name_but_no_payload():
    events: list[dict] = []
    with pytest.raises(HookAnswerException):
        HookRunner([BlockIf(as_=BlockAs.ANSWER), Suffix(tag="Z")]).run(
            HookPoint.ON_INPUT, HookContext(), "SSN 123", events=events
        )
    assert events == [
        {"hook": "BlockIf", "type": "BlockIf", "point": "on_input", "decision": "block", "outcome": "answer"}
    ]
    assert "123" not in str(events)


def test_modify_event_records_sizes_and_whether_it_changed_never_the_content():
    _, events = run([Suffix(tag="!!")], HookPoint.ON_INPUT, "secret")
    assert events[0]["decision"] == "modify" and events[0]["changed"] is True
    assert (events[0]["before_chars"], events[0]["after_chars"]) == (6, 8)
    assert "secret" not in str(events)


def test_modify_that_changes_nothing_is_reported_as_unchanged():
    _, events = run([Suffix(tag="")], HookPoint.ON_INPUT, "same")
    assert events[0]["changed"] is False


def test_skip_is_only_valid_at_before_tool():
    class Cache(Hook):
        def before_tool(self, ctx, call):
            return Skip("cached")

        def on_input(self, ctx, text):
            return Skip("nope")

    result, _ = run([Cache()], HookPoint.BEFORE_TOOL, CALL.input, call=CALL)
    assert result.skip == Skip("cached")
    assert run([Cache()], HookPoint.ON_INPUT, "x")[0].value == "x"


def test_tools_filter_applies_to_tool_points_and_compares_sanitized_names():
    class Deny(Hook):
        def before_tool(self, ctx, call):
            return Block("denied")

    other = ToolCall(name="Other Tool", input={})
    assert run([Deny(tools=["search"])], HookPoint.BEFORE_TOOL, {}, call=other)[0].value == {}
    with pytest.raises(ToolBlockedException):
        run([Deny(tools=["Other-Tool"])], HookPoint.BEFORE_TOOL, {}, call=other)


def test_hook_crash_fails_closed_by_default_and_never_traces_the_exception_text():
    events: list[dict] = []
    with pytest.raises(HookStopException):
        HookRunner([Crash()]).run(HookPoint.ON_INPUT, HookContext(), "x", events=events)
    assert events[0]["decision"] == "error" and events[0]["error_type"] == "RuntimeError"
    assert "secret payload" not in str(events)


def test_hook_crash_at_a_tool_point_becomes_an_observation():
    with pytest.raises(ToolBlockedException):
        run([Crash()], HookPoint.BEFORE_TOOL, {}, call=CALL)


def test_on_error_skip_continues_and_stop_fails():
    assert run([Crash(on_error=HookErrorPolicy.SKIP)], HookPoint.ON_INPUT, "x")[0].value == "x"
    with pytest.raises(HookStopException):
        run([Crash(on_error=HookErrorPolicy.STOP)], HookPoint.BEFORE_TOOL, {}, call=CALL)


def test_timeout_counts_as_a_failure():
    started = time.monotonic()
    with pytest.raises(HookStopException):
        run([Sleepy(timeout_seconds=0.05)], HookPoint.ON_INPUT, "x")
    assert time.monotonic() - started < 0.4
    assert run([Sleepy(timeout_seconds=0.05, on_error=HookErrorPolicy.SKIP)], HookPoint.ON_INPUT, "x")[0].value == "x"


def test_points_default_to_the_overridden_methods():
    assert Suffix().points() == {HookPoint.ON_INPUT, HookPoint.ON_OUTPUT}
    assert Hook().points() == set()
    runner = HookRunner([Suffix()])
    assert runner.has(HookPoint.ON_OUTPUT) and not runner.has(HookPoint.BEFORE_MODEL)


def test_call_views_share_state_and_each_hook_has_its_own_slice():
    ctx = HookContext(metadata={"role": "admin"}, user_id="u1")
    view = ctx.for_call(loop=3, hook_key="0:a")
    view.hook_state["n"] = 1
    assert ctx.state["hooks"]["0:a"] == {"n": 1} and view.loop == 3 and ctx.loop is None
    assert ctx.for_call(hook_key="1:b").hook_state == {}
    assert (ctx.lookup("metadata.role"), ctx.lookup("user_id"), ctx.lookup("metadata.missing")) == ("admin", "u1", None)


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        Suffix(tagg="x")


def test_resolve_requires_a_type_and_names_the_known_ones():
    with pytest.raises(ValueError, match="missing `type`"):
        resolve_hook({"patterns": ["x"]})
    with pytest.raises(ValueError, match="Unknown hook type 'piii'.*pii"):
        resolve_hook({"type": "piii"})
    with pytest.raises(ValueError, match="could not be imported"):
        resolve_hook({"type": "no.such.module.Hook"})
    with pytest.raises(ValueError, match="not a Hook subclass"):
        resolve_hook({"type": "json.JSONDecoder"})


def test_python_subclass_dumps_its_class_path_and_all_its_fields():
    dumped = Suffix(tag="X").model_dump()
    assert dumped["type"].endswith(".Suffix") and dumped["tag"] == "X"
    assert "type" not in Hook().model_dump()


def test_builtin_dump_uses_the_short_type_and_round_trips():
    hook = resolve_hook({"type": "call_limit", "tools": ["search"], "max_per_run": 2})
    dumped = hook.model_dump()
    assert dumped["type"] == "call_limit" and dumped["max_per_run"] == 2
    assert resolve_hook(dumped) == hook


def test_live_answer_filter_never_sends_a_partial_match_whatever_the_chunking():
    from dynamiq.nodes.agents.hooks import PIIHook

    hook = PIIHook(on=["input", "tool_result", "output"], live_stream=True, stream_lookback=24)
    answer = "Write to zq9ann@leak.example or zq9bob@leak.example, call +1 415 555 0100, thanks for waiting."
    expected = PIIHook(on=["output"]).on_output(HookContext(hook_key="k"), answer).value
    for size in (1, 2, 3, 5, 8, 13, len(answer)):
        runner = HookRunner([hook])
        live = runner.live_answer_filter(HookContext())
        sent = ""
        for start in range(0, len(answer), size):
            sent += live.feed(answer[start : start + size])
            assert "zq9" not in sent and "415 555" not in sent, (size, sent)
        sent += live.finish()
        assert sent == expected, size


def test_live_answer_filter_exists_only_when_every_answer_hook_can_stream():
    from dynamiq.nodes.agents.hooks import PIIHook, RegexHook

    ctx = HookContext()
    live = PIIHook(on=["output"], live_stream=True)
    assert HookRunner([live]).live_answer_filter(ctx) is not None
    assert HookRunner([live, Suffix()]).live_answer_filter(ctx) is None  # a Python hook that rewrites the answer
    assert HookRunner([PIIHook()]).live_answer_filter(ctx) is None  # not opted in
    assert HookRunner([RegexHook(patterns=["x"], action="block", live_stream=True)]).live_answer_filter(ctx) is None
