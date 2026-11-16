"""Opt-in prompt compaction on real agent runs.

With the setting on, a ``chat`` span of an agent loop records only the messages its prompt adds to the prompt of the
loop before it. The whole prompt of every span, rebuilt by following ``dynamiq.prompt.messages_base.*``, has to be
what the same run records with the setting off.
"""

import asyncio
import threading

import pytest
from litellm import ModelResponse

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.callbacks.tracing import PROMPT_MESSAGES_BASE_KEY, RunType
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.tools.agent_tool import SubAgentTool
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableStatus
from tests.helpers.agent_hooks import (
    ALL_MODES,
    SearchTool,
    build_agent,
    encode,
    final_step,
    make_llm,
    parallel_step,
    tool_step,
)
from tests.integration.telemetry.conftest import NODE_NAME, RUN_ID, Trace, attrs_json
from tests.integration.telemetry.rebuild import (
    INPUT_MESSAGES,
    MESSAGES_BASE_COUNT,
    MESSAGES_BASE_SPAN_ID,
    expand_input_messages,
    rebuild,
    span_id_of,
)
from tests.integration.telemetry.test_rebuild import TAGS, assert_matches

LOOPS = 6


def loop_steps(loops: int) -> list[tuple]:
    """A script of ``loops`` model calls: every call but the last asks for a tool."""
    return [tool_step("search", query=f"q{index}") for index in range(loops - 1)] + [final_step("done")]


def chats_of(spans: Trace, parent=None) -> list:
    """``chat`` spans in the order their runs started; below ``parent`` when one is given."""
    chats = spans.by_op("chat") if parent is None else [s for s in spans.children(parent) if s in spans.by_op("chat")]
    return sorted(chats, key=lambda span: span.start_time)


def has_base(span) -> bool:
    assert (MESSAGES_BASE_SPAN_ID in span.attributes) == (MESSAGES_BASE_COUNT in span.attributes)
    return MESSAGES_BASE_SPAN_ID in span.attributes


@pytest.fixture
def run_loops(script_llm, make_config, exporter, trace):
    """Runs a new agent for ``loops`` model calls and returns the spans of that run alone."""

    def run(mode: InferenceMode, loops: int = LOOPS, run_async: bool = False, **tracing) -> Trace:
        exporter.clear()
        agent = build_agent(tools=[SearchTool(result="forty-two")], inference_mode=mode, max_loops=loops + 2)
        script_llm(mode, loop_steps(loops))
        config = make_config(**tracing)
        if run_async:
            result = asyncio.run(agent.run_async(input_data={"input": "hello"}, config=config))
        else:
            result = agent.run(input_data={"input": "hello"}, config=config)
        assert result.output["content"] == "done"
        return trace()

    return run


@pytest.mark.parametrize("mode", ALL_MODES, ids=lambda m: m.value)
def test_agent_loop_exports_each_message_once_and_expands_to_every_whole_prompt(mode, run_loops):
    off = run_loops(mode)
    whole_prompts = [attrs_json(span, INPUT_MESSAGES) for span in chats_of(off)]
    assert len(whole_prompts) == LOOPS
    assert not any(has_base(span) for span in off.spans)

    on = run_loops(mode, prompt_compaction_enabled=True)

    chats = chats_of(on)
    assert len(on.spans) == len(off.spans)
    # following the pointers gives back every prompt, message for message
    assert [expand_input_messages(on.spans, span) for span in chats] == whole_prompts
    # the conversation is exported once: what the spans record, one after the other, is the last prompt
    recorded = [message for span in chats for message in attrs_json(span, INPUT_MESSAGES)]
    assert recorded == whole_prompts[-1]
    # the first loop holds its whole prompt; every later one continues all of the prompt before it
    assert not has_base(chats[0])
    for previous, span, previous_prompt in zip(chats, chats[1:], whole_prompts):
        assert span.attributes[MESSAGES_BASE_SPAN_ID] == span_id_of(previous)
        assert span.attributes[MESSAGES_BASE_COUNT] == len(previous_prompt)
    # only chat spans change, and only by the two attributes
    for span, twin in zip(chats, chats_of(off)):
        assert set(span.attributes) - {MESSAGES_BASE_SPAN_ID, MESSAGES_BASE_COUNT} == set(twin.attributes)
    assert not any(has_base(span) for span in on.spans if span not in chats)


def test_compacted_trace_holds_far_fewer_message_bytes(run_loops):
    mode = InferenceMode.FUNCTION_CALLING

    def message_bytes(spans: Trace) -> int:
        return sum(len(span.attributes[INPUT_MESSAGES]) for span in chats_of(spans))

    off = run_loops(mode, loops=12)
    last_prompt = len(chats_of(off)[-1].attributes[INPUT_MESSAGES])
    off_bytes = message_bytes(off)

    on_bytes = message_bytes(run_loops(mode, loops=12, prompt_compaction_enabled=True))

    # Off, every span repeats the conversation so far; on, the spans together hold it once (plus the brackets of
    # twelve JSON lists instead of one).
    assert off_bytes > 6 * last_prompt
    assert last_prompt <= on_bytes < last_prompt + 12 * 4


def test_async_agent_run_is_compacted_the_same_way(run_loops):
    mode = InferenceMode.FUNCTION_CALLING
    whole_prompts = [attrs_json(span, INPUT_MESSAGES) for span in chats_of(run_loops(mode, run_async=True))]

    on = run_loops(mode, run_async=True, prompt_compaction_enabled=True)

    chats = chats_of(on)
    assert [has_base(span) for span in chats] == [False] + [True] * (LOOPS - 1)
    assert [expand_input_messages(on.spans, span) for span in chats] == whole_prompts


def test_setting_off_on_the_config_wins_over_the_environment(run_loops, monkeypatch):
    monkeypatch.setenv("DYNAMIQ_TRACING_PROMPT_COMPACTION_ENABLED", "true")

    by_environment = run_loops(InferenceMode.XML, loops=3)
    assert [has_base(span) for span in chats_of(by_environment)] == [False, True, True]

    off = run_loops(InferenceMode.XML, loops=3, prompt_compaction_enabled=False)
    assert not any(has_base(span) for span in off.spans)


def test_content_capture_off_leaves_no_pointer(run_loops):
    from dynamiq.types.telemetry import ContentCaptureMode

    spans = run_loops(
        InferenceMode.FUNCTION_CALLING, prompt_compaction_enabled=True, content_capture=ContentCaptureMode.NO_CONTENT
    )

    chats = chats_of(spans)
    assert len(chats) == LOOPS
    for span in spans.spans:
        assert INPUT_MESSAGES not in span.attributes
        assert not has_base(span)


# Several parent runs: a sub-agent called twice in one step


@pytest.fixture
def script_by_prompt(mocker):
    """Makes every LLM node answer from its own request alone, so that agents running at once cannot take each
    other's step: an agent that has the ``search`` tool follows ``worker_steps``, any other ``lead_steps``, at the
    step numbered by the tool results its prompt already holds.

    With ``workers_at_once``, the first model call of a worker waits until that many workers are in theirs, so
    that the worker runs are known to overlap.
    """

    def install(lead_steps: list[tuple], worker_steps: list[tuple], workers_at_once: int = 0) -> None:
        mode = InferenceMode.FUNCTION_CALLING
        all_workers_started = threading.Barrier(workers_at_once) if workers_at_once else None

        def completion(stream=False, *args, **kwargs):
            tool_names = {tool["function"]["name"] for tool in kwargs.get("tools") or []}
            is_worker = "search" in tool_names
            steps = worker_steps if is_worker else lead_steps
            observations = sum(1 for message in kwargs["messages"] if message.get("role") == "tool")
            if is_worker and observations == 0 and all_workers_started is not None:
                all_workers_started.wait(timeout=30)
            output = encode(mode, steps[min(observations, len(steps) - 1)], observations)
            return ModelResponse(
                choices=[
                    {"message": {"content": "", "tool_calls": output["tool_calls"]}, "finish_reason": "tool_calls"}
                ]
            )

        mocker.patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=completion)

    return install


@pytest.fixture
def run_lead_with_two_worker_runs(script_by_prompt, exporter, trace):
    """Runs a lead agent that calls its sub-agent twice in one step and returns the spans of that run alone.

    ``at_once`` decides how the sub-agent runs twice. False: one agent object runs twice, one run after the other,
    and shares its LLM node with the lead, so nothing but the parent run tells the eight model calls apart. True:
    every call runs a new agent, and the two run at the same time.
    """

    def run(config, at_once: bool) -> Trace:
        exporter.clear()
        mode = InferenceMode.FUNCTION_CALLING
        llm = make_llm()

        def worker() -> Agent:
            return Agent(
                id="worker",
                name="Worker",
                llm=make_llm() if at_once else llm,
                role="research",
                tools=[SearchTool()],
                inference_mode=mode,
                max_loops=6,
            )

        how = {"agent_factory": worker} if at_once else {"agent": worker()}
        lead = Agent(
            id="lead",
            name="Lead",
            llm=llm,
            role="delegate",
            tools=[SubAgentTool(name="Worker", description="Researches.", **how)],
            inference_mode=mode,
            parallel_tool_calls_enabled=True,
            max_loops=6,
        )
        script_by_prompt(
            lead_steps=[
                parallel_step(("Worker", {"input": "dig a"}), ("Worker", {"input": "dig b"})),
                final_step("all"),
            ],
            worker_steps=[tool_step("search", query="one"), tool_step("search", query="two"), final_step("found")],
            workers_at_once=2 if at_once else 0,
        )

        result = lead.run(input_data={"input": "hello"}, config=config)

        assert result.output["content"] == "all"
        return trace()

    return run


def prompts_by_agent_run(spans: Trace, expand: bool) -> dict[tuple[str, str], list]:
    """``(agent name, agent input) -> whole prompts of its model calls``, in the order of the calls."""
    prompts = {}
    for agent in spans.by_op("invoke_agent"):
        key = (agent.attributes[NODE_NAME], attrs_json(agent, INPUT_MESSAGES)[0]["parts"][0]["content"])
        assert key not in prompts
        prompts[key] = [
            expand_input_messages(spans.spans, chat) if expand else attrs_json(chat, INPUT_MESSAGES)
            for chat in chats_of(spans, agent)
        ]
    return prompts


@pytest.mark.parametrize("at_once", [False, True], ids=["one-llm-node-in-turn", "new-agents-at-once"])
def test_runs_of_a_sub_agent_keep_their_own_chains(run_lead_with_two_worker_runs, make_config, at_once):
    off = run_lead_with_two_worker_runs(make_config(), at_once)
    whole_prompts = prompts_by_agent_run(off, expand=False)
    assert {key: len(prompts) for key, prompts in whole_prompts.items()} == {
        ("Lead", "hello"): 2,
        ("Worker", "dig a"): 3,
        ("Worker", "dig b"): 3,
    }

    on = run_lead_with_two_worker_runs(make_config(prompt_compaction_enabled=True), at_once)

    if not at_once:
        assert len({chat.attributes["dynamiq.node.id"] for chat in on.by_op("chat")}) == 1
    by_span_id = {span_id_of(span): span for span in on.spans}
    for agent in on.by_op("invoke_agent"):
        chats = chats_of(on, agent)
        # the first call of every agent run holds its whole prompt, whatever its LLM node recorded elsewhere
        assert not has_base(chats[0])
        for previous, chat in zip(chats, chats[1:]):
            base = by_span_id[chat.attributes[MESSAGES_BASE_SPAN_ID]]
            assert base is previous
            assert on.parent_of(base) is agent
    assert prompts_by_agent_run(on, expand=True) == whole_prompts


# The run-based projection


def test_compacted_trace_rebuilds_into_the_runs_a_compacting_handler_recorded(script_llm, make_config, trace):
    mode = InferenceMode.FUNCTION_CALLING
    handler = TracingCallbackHandler(
        tags=TAGS, source_id="source-1", session_id="session-1", compact_llm_prompts_enabled=True
    )
    config = make_config(
        callbacks=[handler], tags=TAGS, source_id="source-1", session_id="session-1", prompt_compaction_enabled=True
    )
    agent = build_agent(tools=[SearchTool(result="forty-two")], inference_mode=mode)
    script_llm(mode, loop_steps(4))

    result = Workflow(id="wf", flow=Flow(id="flow", nodes=[agent])).run(input_data={"input": "hello"}, config=config)

    assert result.status == RunnableStatus.SUCCESS
    spans = trace().spans
    rebuilt = rebuild(spans)
    # every field of the "Rebuilding a run" table, the compacted prompt messages among them
    assert_matches(rebuilt, handler, spans)
    llm_runs = [
        run for run in handler.runs.values() if run.type == RunType.NODE and run.metadata["node"]["group"] == "llms"
    ]
    assert len(llm_runs) == 4
    bases = []
    for run in llm_runs:
        base = run.metadata["node"]["prompt"][PROMPT_MESSAGES_BASE_KEY]
        bases.append(base)
        expected = None if base is None else {"run_id": str(base["run_id"]), "count": base["count"]}
        assert rebuilt[str(run.id)].metadata["node"]["prompt"]["messages_base"] == expected
    assert [base is None for base in bases] == [True, False, False, False]
    # the pointer names a run by its id, which is the run id of the span the attribute names
    by_run_id = {span.attributes[RUN_ID]: span for span in spans}
    for run in llm_runs[1:]:
        base_run_id = rebuilt[str(run.id)].metadata["node"]["prompt"]["messages_base"]["run_id"]
        assert by_run_id[str(run.id)].attributes[MESSAGES_BASE_SPAN_ID] == span_id_of(by_run_id[base_run_id])
