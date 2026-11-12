"""Opt-in prompt compaction: a ``chat`` span that continues the prompt of the previous ``chat`` span of the same LLM
node under the same parent run records only the messages that follow the shared start.

The first tests are the span twins of tests/unit/callbacks/test_tracing_prompt_compaction.py, one per case, in the
same order: ``TracingCallbackHandler`` and the spans follow the same rules.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest

from dynamiq.nodes import NodeGroup
from dynamiq.nodes.node import ErrorHandling
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.telemetry import registry, semconv, settings
from dynamiq.types.telemetry import ContentCaptureMode, TracingConfig
from tests.unit.telemetry.conftest import Parent, Prompting, parent_span_id

sdk_trace = pytest.importorskip("opentelemetry.sdk.trace")
sampling = pytest.importorskip("opentelemetry.sdk.trace.sampling")

pytestmark = pytest.mark.usefixtures("no_span_leaks")

INPUT_MESSAGES = "gen_ai.input.messages"
BASE_SPAN_ID = "dynamiq.prompt.messages_base.span_id"
BASE_COUNT = "dynamiq.prompt.messages_base.count"
COMPACTION_VARIABLE = "DYNAMIQ_TRACING_PROMPT_COMPACTION_ENABLED"

SYSTEM = {"role": "system", "content": "You are an agent with a long system prompt."}
USER = {"role": "user", "content": "Question"}
STEP_1 = {"role": "assistant", "content": "Thought 1"}
OBSERVATION_1 = {"role": "user", "content": "Observation 1"}
STEP_2 = {"role": "assistant", "content": "Thought 2"}
OBSERVATION_2 = {"role": "user", "content": "Observation 2"}

RETRY_ONCE = ErrorHandling(max_retries=1, retry_interval_seconds=0.01)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    monkeypatch.delenv(COMPACTION_VARIABLE, raising=False)
    monkeypatch.setattr(settings, "_warned_variables", set())


@pytest.fixture
def compacting(provider) -> RunnableConfig:
    return RunnableConfig(tracing=TracingConfig(tracer_provider=provider, prompt_compaction_enabled=True))


def run_parent(config: RunnableConfig, script) -> None:
    """Run ``script(config, parent_run_id)`` inside a node run, which is the parent run of what the script runs."""
    parent = Parent()
    parent._script = script
    assert parent.run(input_data={}, config=config).status == RunnableStatus.SUCCESS


def run_llm(config: RunnableConfig, parent_run_id, messages, node_id: str = "llm-1", llm: Prompting | None = None):
    llm = llm or Prompting(id=node_id)
    return llm.run(input_data={"messages": messages}, config=config, parent_run_id=parent_run_id)


def chat_spans(exporter) -> list:
    """The spans of the LLM runs, in the order the runs started."""
    spans = [span for span in exporter.get_finished_spans() if span.attributes.get("gen_ai.operation.name") == "chat"]
    return sorted(spans, key=lambda span: span.start_time)


def span_id_of(span) -> str:
    return format(span.context.span_id, "016x")


def parts(messages: list[dict]) -> list[dict]:
    """Text messages as ``gen_ai.input.messages`` records them."""
    return [{"role": m["role"], "parts": [{"type": "text", "content": m["content"]}]} for m in messages]


def base_of(span) -> dict | None:
    """``{"span_id", "count"}`` of a compacted span, None for a span that holds its whole prompt."""
    attributes = span.attributes
    assert (BASE_SPAN_ID in attributes) == (BASE_COUNT in attributes)
    if BASE_SPAN_ID not in attributes:
        return None
    return {"span_id": attributes[BASE_SPAN_ID], "count": attributes[BASE_COUNT]}


def recorded(span) -> dict:
    return {"messages": json.loads(span.attributes[INPUT_MESSAGES]), "base": base_of(span)}


def full_messages(spans: list, span) -> list[dict]:
    """The whole prompt of a span: the first ``count`` messages of its base's whole prompt, then its own."""
    own = json.loads(span.attributes[INPUT_MESSAGES])
    base = base_of(span)
    if base is None:
        return own
    (base_span,) = (candidate for candidate in spans if span_id_of(candidate) == base["span_id"])
    return full_messages(spans, base_span)[: base["count"]] + own


# Twins of the TracingCallbackHandler cases


def test_iterations_record_only_new_messages(exporter, compacting):
    prompts = [
        [SYSTEM, USER],
        [SYSTEM, USER, STEP_1, OBSERVATION_1],
        [SYSTEM, USER, STEP_1, OBSERVATION_1, STEP_2, OBSERVATION_2],
    ]

    run_parent(compacting, lambda config, parent: [run_llm(config, parent, prompt) for prompt in prompts])

    spans = chat_spans(exporter)
    assert recorded(spans[0]) == {"messages": parts([SYSTEM, USER]), "base": None}
    assert recorded(spans[1]) == {
        "messages": parts([STEP_1, OBSERVATION_1]),
        "base": {"span_id": span_id_of(spans[0]), "count": 2},
    }
    assert recorded(spans[2]) == {
        "messages": parts([STEP_2, OBSERVATION_2]),
        "base": {"span_id": span_id_of(spans[1]), "count": 4},
    }
    for span, prompt in zip(spans, prompts):
        assert full_messages(spans, span) == parts(prompt)


def test_rewritten_history_keeps_only_the_shared_start(exporter, compacting):
    summary = {"role": "user", "content": "Summary of earlier steps"}

    def script(config, parent):
        run_llm(config, parent, [SYSTEM, USER, STEP_1, OBSERVATION_1])
        run_llm(config, parent, [SYSTEM, summary])

    run_parent(compacting, script)

    first, second = chat_spans(exporter)
    assert recorded(second) == {"messages": parts([summary]), "base": {"span_id": span_id_of(first), "count": 1}}
    assert full_messages([first, second], second) == parts([SYSTEM, summary])


def test_prompt_without_shared_start_is_recorded_whole(exporter, compacting):
    def script(config, parent):
        run_llm(config, parent, [SYSTEM, USER])
        run_llm(config, parent, [USER])

    run_parent(compacting, script)

    assert recorded(chat_spans(exporter)[1]) == {"messages": parts([USER]), "base": None}


def test_runs_of_other_parents_and_llms_are_not_bases(exporter, compacting):
    run_parent(compacting, lambda config, parent: run_llm(config, parent, [SYSTEM, USER]))

    def script(config, parent):
        run_llm(config, parent, [SYSTEM, USER], node_id="llm-2")
        run_llm(config, parent, [SYSTEM, USER, STEP_1])

    run_parent(compacting, script)

    assert recorded(chat_spans(exporter)[2]) == {"messages": parts([SYSTEM, USER, STEP_1]), "base": None}


def test_retried_run_is_not_its_own_base(exporter, compacting):
    retried = Prompting(id="llm-1", fail_times=1, error_handling=RETRY_ONCE)

    def script(config, parent):
        run_llm(config, parent, [SYSTEM, USER])
        # Reports its prompt twice: once per attempt.
        assert run_llm(config, parent, [SYSTEM, USER, STEP_1], llm=retried).status == RunnableStatus.SUCCESS

    run_parent(compacting, script)

    first, second = chat_spans(exporter)
    assert retried._attempts == 2
    assert recorded(second) == {"messages": parts([STEP_1]), "base": {"span_id": span_id_of(first), "count": 2}}


def test_retry_between_concurrent_runs_keeps_what_was_recorded(exporter, compacting):
    other_user = {"role": "user", "content": "Other question"}
    retried = Prompting(id="llm-1", fail_times=1, error_handling=RETRY_ONCE)

    def script(config, parent):
        # The other run reports between the two attempts of the retried one, and takes it as its base.
        retried._before_failing = lambda: run_llm(config, parent, [SYSTEM, other_user])
        run_llm(config, parent, [SYSTEM, USER], llm=retried)

    run_parent(compacting, script)

    first, second = chat_spans(exporter)
    assert retried._attempts == 2
    assert recorded(first) == {"messages": parts([SYSTEM, USER]), "base": None}
    assert base_of(second) == {"span_id": span_id_of(first), "count": 1}
    assert full_messages([first, second], second) == parts([SYSTEM, other_user])


def test_compaction_is_off_by_default(exporter, traced):
    def script(config, parent):
        run_llm(config, parent, [SYSTEM, USER])
        run_llm(config, parent, [SYSTEM, USER, STEP_1])

    run_parent(traced, script)

    assert recorded(chat_spans(exporter)[1]) == {"messages": parts([SYSTEM, USER, STEP_1]), "base": None}


# Which runs take part


def test_run_without_a_parent_run_is_never_compacted(exporter, compacting):
    llm = Prompting(id="llm-1")

    llm.run(input_data={"messages": [SYSTEM, USER]}, config=compacting)
    llm.run(input_data={"messages": [SYSTEM, USER, STEP_1]}, config=compacting)

    assert [recorded(span) for span in chat_spans(exporter)] == [
        {"messages": parts([SYSTEM, USER]), "base": None},
        {"messages": parts([SYSTEM, USER, STEP_1]), "base": None},
    ]


def test_run_whose_parent_run_has_no_span_in_this_process_is_not_compacted(exporter, compacting):
    # A node that a worker process runs names the flow run of the parent process, which has no span here to keep
    # the prompt of the previous run.
    parent_run_id = uuid4()

    run_llm(compacting, parent_run_id, [SYSTEM, USER])
    run_llm(compacting, parent_run_id, [SYSTEM, USER, STEP_1])

    assert [base_of(span) for span in chat_spans(exporter)] == [None, None]
    assert recorded(chat_spans(exporter)[1])["messages"] == parts([SYSTEM, USER, STEP_1])


def test_next_run_of_the_parent_starts_again_from_a_whole_prompt(exporter, compacting):
    # What a parent run keeps of its LLM runs goes with it.
    for _ in range(2):
        run_parent(compacting, lambda config, parent: run_llm(config, parent, [SYSTEM, USER]))

    assert [recorded(span) for span in chat_spans(exporter)] == [{"messages": parts([SYSTEM, USER]), "base": None}] * 2


def test_identical_prompt_records_no_message_of_its_own(exporter, compacting):
    def script(config, parent):
        run_llm(config, parent, [SYSTEM, USER])
        run_llm(config, parent, [SYSTEM, USER])

    run_parent(compacting, script)

    first, second = chat_spans(exporter)
    assert recorded(second) == {"messages": [], "base": {"span_id": span_id_of(first), "count": 2}}
    assert full_messages([first, second], second) == parts([SYSTEM, USER])


def test_empty_prompt_takes_no_part(exporter, compacting):
    # The handler ignores a report without messages, so the run before it stays the one to compare with.
    prompts = [[SYSTEM, USER], [], [SYSTEM, USER, STEP_1]]

    run_parent(compacting, lambda config, parent: [run_llm(config, parent, prompt) for prompt in prompts])

    first, empty, third = chat_spans(exporter)
    assert recorded(empty) == {"messages": [], "base": None}
    assert recorded(third) == {"messages": parts([STEP_1]), "base": {"span_id": span_id_of(first), "count": 2}}


def test_only_chat_spans_are_compacted(exporter, compacting):
    def script(config, parent):
        for prompt in ([SYSTEM, USER], [SYSTEM, USER, STEP_1]):
            run_llm(config, parent, prompt, llm=Prompting(id="not-an-llm", name="other", group=NodeGroup.UTILS))

    run_parent(compacting, script)

    spans = sorted((s for s in exporter.get_finished_spans() if s.name == "other"), key=lambda s: s.start_time)
    assert [recorded(span) for span in spans] == [
        {"messages": parts([SYSTEM, USER]), "base": None},
        {"messages": parts([SYSTEM, USER, STEP_1]), "base": None},
    ]


def test_base_is_always_a_run_of_the_same_parent_when_parents_run_at_once(exporter, compacting):
    loops = 6

    def prompts_of(index: int) -> list[list[dict]]:
        question = {"role": "user", "content": f"Question {index}"}
        steps = [{"role": "assistant", "content": f"Thought {index}.{loop}"} for loop in range(loops)]
        return [[SYSTEM, question, *steps[:loop]] for loop in range(loops)]

    def run(index: int) -> None:
        # Every parent runs an LLM node with the same id, as every run of one agent does.
        run_parent(compacting, lambda config, parent: [run_llm(config, parent, p) for p in prompts_of(index)])

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(run, range(16)))

    spans = chat_spans(exporter)
    by_span_id = {span_id_of(span): span for span in spans}
    by_parent: dict[int, list] = {}
    for span in spans:
        by_parent.setdefault(parent_span_id(span), []).append(span)
    assert len(by_parent) == 16
    for siblings in by_parent.values():
        index = int(json.loads(siblings[0].attributes[INPUT_MESSAGES])[1]["parts"][0]["content"].split()[-1])
        assert [full_messages(spans, span) for span in siblings] == [parts(p) for p in prompts_of(index)]
        assert base_of(siblings[0]) is None
        for previous, span in zip(siblings, siblings[1:]):
            assert base_of(span)["span_id"] == span_id_of(previous)
            assert parent_span_id(by_span_id[base_of(span)["span_id"]]) == parent_span_id(span)


def test_message_that_is_not_recorded_is_not_counted(exporter, compacting):
    # ``count`` is a number of recorded messages: the first ``count`` messages of the base span's whole prompt.
    def script(config, parent):
        run_llm(config, parent, [SYSTEM, None, USER])
        run_llm(config, parent, [SYSTEM, None, USER, STEP_1])

    run_parent(compacting, script)

    first, second = chat_spans(exporter)
    assert recorded(first) == {"messages": parts([SYSTEM, USER]), "base": None}
    assert recorded(second) == {"messages": parts([STEP_1]), "base": {"span_id": span_id_of(first), "count": 2}}


# What is never a base


def test_content_capture_off_records_neither_messages_nor_a_base(exporter, provider, mocker):
    keep = mocker.spy(registry, "keep_prompt")
    config = RunnableConfig(
        tracing=TracingConfig(
            tracer_provider=provider, prompt_compaction_enabled=True, content_capture=ContentCaptureMode.NO_CONTENT
        )
    )

    def script(config, parent):
        run_llm(config, parent, [SYSTEM, USER])
        run_llm(config, parent, [SYSTEM, USER, STEP_1])

    run_parent(config, script)

    spans = chat_spans(exporter)
    assert len(spans) == 2
    for span in spans:
        assert INPUT_MESSAGES not in span.attributes
        assert BASE_SPAN_ID not in span.attributes
        assert BASE_COUNT not in span.attributes
    # No prompt is kept for a comparison that would record nothing.
    assert keep.call_count == 0


@pytest.mark.parametrize("decision", ["DROP", "RECORD_ONLY"])
def test_span_that_is_not_exported_is_never_a_base(exporter, decision):
    # A pointer to a span the backend never receives could not be followed.
    class FirstChatNotExported(sampling.Sampler):
        def __init__(self):
            self.chats = 0

        def should_sample(self, parent_context, trace_id, name, kind=None, attributes=None, links=None, **kwargs):
            result = sampling.Decision.RECORD_AND_SAMPLE
            if (attributes or {}).get("gen_ai.operation.name") == "chat":
                self.chats += 1
                if self.chats == 1:
                    result = getattr(sampling.Decision, decision)
            return sampling.SamplingResult(result, attributes)

        def get_description(self) -> str:
            return "FirstChatNotExported"

    provider = sdk_trace.TracerProvider(sampler=FirstChatNotExported())
    provider.add_span_processor(sdk_trace.export.SimpleSpanProcessor(exporter))
    config = RunnableConfig(tracing=TracingConfig(tracer_provider=provider, prompt_compaction_enabled=True))
    prompts = [[SYSTEM, USER], [SYSTEM, USER, STEP_1], [SYSTEM, USER, STEP_1, OBSERVATION_1]]

    run_parent(config, lambda config, parent: [run_llm(config, parent, prompt) for prompt in prompts])

    second, third = chat_spans(exporter)
    assert recorded(second) == {"messages": parts([SYSTEM, USER, STEP_1]), "base": None}
    assert recorded(third) == {"messages": parts([OBSERVATION_1]), "base": {"span_id": span_id_of(second), "count": 3}}


def test_prompt_that_cannot_be_recorded_is_not_a_base(exporter, compacting):
    not_a_message = 5
    prompts = [[SYSTEM, USER], [SYSTEM, not_a_message], [SYSTEM, USER, STEP_1]]

    run_parent(compacting, lambda config, parent: [run_llm(config, parent, prompt) for prompt in prompts])

    first, broken, third = chat_spans(exporter)
    assert INPUT_MESSAGES not in broken.attributes
    assert base_of(broken) is None
    assert broken.attributes["dynamiq.run.status"] == "succeeded"
    # The run after it continues the last prompt that was recorded.
    assert recorded(third) == {"messages": parts([STEP_1]), "base": {"span_id": span_id_of(first), "count": 2}}


def test_prompt_mapping_that_raises_costs_neither_the_run_nor_the_span(exporter, compacting, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("mapping bug")

    monkeypatch.setattr(semconv, "prompt_attributes", broken)
    results = []

    def script(config, parent):
        results.extend(run_llm(config, parent, prompt) for prompt in ([SYSTEM, USER], [SYSTEM, USER, STEP_1]))

    run_parent(compacting, script)

    assert [result.status for result in results] == [RunnableStatus.SUCCESS] * 2
    assert [span.attributes["dynamiq.run.status"] for span in chat_spans(exporter)] == ["succeeded"] * 2


# The setting


def resolved(**tracing_fields) -> bool:
    return settings.resolve_settings(RunnableConfig(tracing=TracingConfig(**tracing_fields))).prompt_compaction_enabled


def test_setting_is_off_unless_the_config_or_the_environment_turns_it_on(monkeypatch):
    assert settings.resolve_settings(None).prompt_compaction_enabled is False
    assert resolved() is False
    assert resolved(prompt_compaction_enabled=True) is True

    monkeypatch.setenv(COMPACTION_VARIABLE, "true")

    assert settings.resolve_settings(None).prompt_compaction_enabled is True
    assert resolved() is True
    # A field set on the config wins over the environment, as for the other capture settings.
    assert resolved(prompt_compaction_enabled=False) is False


def test_setting_that_is_neither_true_nor_false_is_off_and_warns_once(monkeypatch, caplog):
    monkeypatch.setenv(COMPACTION_VARIABLE, "sometimes")

    with caplog.at_level(logging.WARNING):
        assert resolved() is False
        assert resolved() is False

    assert [record.getMessage() for record in caplog.records if COMPACTION_VARIABLE in record.getMessage()] == [
        f"{COMPACTION_VARIABLE}='sometimes' is neither true nor false; using false."
    ]


def test_environment_turns_compaction_on_for_a_run(exporter, traced, monkeypatch):
    monkeypatch.setenv(COMPACTION_VARIABLE, "true")

    def script(config, parent):
        run_llm(config, parent, [SYSTEM, USER])
        run_llm(config, parent, [SYSTEM, USER, STEP_1])

    run_parent(traced, script)

    first, second = chat_spans(exporter)
    assert recorded(second) == {"messages": parts([STEP_1]), "base": {"span_id": span_id_of(first), "count": 2}}
