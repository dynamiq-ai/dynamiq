"""Test helpers over ``dynamiq.telemetry.rebuild``, keyed by run id for the span tests."""

import json
from typing import Any

from dynamiq.telemetry import attributes as attrs
from dynamiq.telemetry.rebuild import RebuiltRun, rebuild_runs

INPUT_MESSAGES = attrs.GEN_AI_INPUT_MESSAGES
MESSAGES_BASE_SPAN_ID = attrs.DYNAMIQ_PROMPT_MESSAGES_BASE_SPAN_ID
MESSAGES_BASE_COUNT = attrs.DYNAMIQ_PROMPT_MESSAGES_BASE_COUNT


def span_id_of(span) -> str:
    """The id of a span as attributes name it: 16 lowercase hex digits."""
    return format(span.context.span_id, "016x")


def expand_input_messages(spans: list, span) -> list[dict[str, Any]]:
    """The whole ``gen_ai.input.messages`` of a ``chat`` span that may be compacted: the first ``count`` messages
    of the whole prompt of the span its ``messages_base`` names, followed by its own."""
    own = json.loads(span.attributes[INPUT_MESSAGES])
    if MESSAGES_BASE_SPAN_ID not in span.attributes:
        return own
    (base,) = (s for s in spans if span_id_of(s) == span.attributes[MESSAGES_BASE_SPAN_ID])
    return expand_input_messages(spans, base)[: span.attributes[MESSAGES_BASE_COUNT]] + own


def rebuild(spans: list, prompt_compaction_enabled: bool | None = None) -> dict[str, RebuiltRun]:
    """Run id -> rebuilt run. Prompt compaction counts as on when a span of the trace was compacted."""
    if prompt_compaction_enabled is None:
        prompt_compaction_enabled = any(MESSAGES_BASE_SPAN_ID in span.attributes for span in spans)
    runs = rebuild_runs(spans, prompt_compaction_enabled=prompt_compaction_enabled)
    return {run["id"]: RebuiltRun.model_validate(run) for run in runs}
