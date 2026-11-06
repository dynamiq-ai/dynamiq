"""A live matrix of agent scenarios across models. It spends money, so it runs only on request.

Set RUN_MODEL_MATRIX=1 and OPENROUTER_API_KEY to run it; without RUN_MODEL_MATRIX nothing here is even
collected, so the default and credentials suites never pay for it. Choose the models with
MODEL_MATRIX_TIER (cheap by default, or flagship) or MODEL_MATRIX_MODELS (comma-separated OpenRouter ids).
MODEL_MATRIX_BUDGET_USD (3 by default) stops the run once that much is spent: the remaining cells skip.

Every cell records its outcome, LLM calls, tokens and cost to MODEL_MATRIX_RESULTS
(model_matrix_results.jsonl by default), one JSON object per line. The file is replaced at the start of each
run, and the budget is checked against it, so it holds across xdist workers.
"""

import json
import os
import time
from pathlib import Path

import pytest
import yaml

from dynamiq.callbacks import BaseCallbackHandler
from dynamiq.nodes.llms.base import BaseLLM

if os.getenv("RUN_MODEL_MATRIX") != "1":
    collect_ignore_glob = ["test_*.py"]

MODELS_FILE = Path(__file__).with_name("models.yaml")
RESULTS_FILE = Path(os.getenv("MODEL_MATRIX_RESULTS", "model_matrix_results.jsonl"))
BUDGET_USD = float(os.getenv("MODEL_MATRIX_BUDGET_USD", "3"))
# Used when a response carries no cost, as for a model missing from litellm's price map; set on the high side.
FALLBACK_USD_PER_MILLION_TOKENS = float(os.getenv("MODEL_MATRIX_FALLBACK_USD_PER_MTOK", "5"))


def selected_models() -> list[dict]:
    catalog = yaml.safe_load(MODELS_FILE.read_text())
    if names := os.getenv("MODEL_MATRIX_MODELS"):
        known = {model["id"]: model for tier in catalog.values() for model in tier}
        return [known.get(name.strip(), {"id": name.strip()}) for name in names.split(",") if name.strip()]
    return catalog[os.getenv("MODEL_MATRIX_TIER", "cheap")]


def pytest_generate_tests(metafunc):
    if "model" in metafunc.fixturenames:
        # One model's cells share a worker (with --dist loadgroup), so its rate limit sees one caller at a time.
        params = [pytest.param(m, id=m["id"], marks=pytest.mark.xdist_group(m["id"])) for m in selected_models()]
        metafunc.parametrize("model", params)


def pytest_sessionstart(session):
    if not hasattr(session.config, "workerinput"):  # the controller, or a run without xdist
        RESULTS_FILE.unlink(missing_ok=True)


def spent_usd() -> float:
    """What the cells finished so far in this run cost, read from the results every worker appends to."""
    try:
        lines = RESULTS_FILE.read_text().splitlines()
    except FileNotFoundError:
        return 0.0
    return sum(json.loads(line).get("cost_usd") or 0 for line in lines if line.strip())


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when == "call":
        item.matrix_report = report


class UsageMeter(BaseCallbackHandler):
    """Adds up the LLM calls, tokens and cost of one cell."""

    def __init__(self):
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost_usd = 0.0

    def on_node_execute_run(self, serialized: dict, **kwargs):
        usage = kwargs.get("usage_data")
        if not usage:
            return
        self.calls += 1
        self.prompt_tokens += usage.get("prompt_tokens") or 0
        self.completion_tokens += usage.get("completion_tokens") or 0
        cost = usage.get("total_tokens_cost_usd")
        # litellm reports 0, not None, for a model missing from its price map, which would leave the budget unspent.
        if not cost:
            cost = (usage.get("total_tokens") or 0) * FALLBACK_USD_PER_MILLION_TOKENS / 1_000_000
        self.cost_usd += cost


class RequestContract:
    """Records every request sent to the provider and the protocol rules it broke.

    A tool reply must answer a call an earlier assistant message made, and an assistant message's calls must
    all be answered before the next assistant message: providers reject either with a 400.
    """

    def __init__(self):
        self.requests: list[dict] = []
        self.violations: list[str] = []

    def check(self, params: dict) -> None:
        self.requests.append(params)
        announced: set[str] = set()
        unanswered: set[str] = set()
        for index, message in enumerate(params.get("messages") or []):
            role = message.get("role")
            if role == "assistant":
                if unanswered:
                    self.violations.append(f"request {len(self.requests)}: calls {sorted(unanswered)} unanswered")
                ids = {call.get("id") for call in message.get("tool_calls") or []}
                announced |= ids
                unanswered = set(ids)
            elif role == "tool":
                call_id = message.get("tool_call_id")
                if call_id not in announced:
                    self.violations.append(f"request {len(self.requests)}: reply {call_id} at {index} has no call")
                unanswered.discard(call_id)
        if unanswered:
            self.violations.append(f"request {len(self.requests)}: calls {sorted(unanswered)} unanswered at the end")


@pytest.fixture
def contract(monkeypatch):
    recorder = RequestContract()
    call_completion, acall_completion = BaseLLM._call_completion, BaseLLM._acall_completion

    def checked(self, params):
        recorder.check(params)
        return call_completion(self, params)

    async def achecked(self, params):
        recorder.check(params)
        return await acall_completion(self, params)

    monkeypatch.setattr(BaseLLM, "_call_completion", checked)
    monkeypatch.setattr(BaseLLM, "_acall_completion", achecked)
    yield recorder
    assert not recorder.violations, recorder.violations


@pytest.fixture
def meter(request, model):
    if not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY is not set")
    spent = spent_usd()
    if spent >= BUDGET_USD:
        pytest.skip(f"budget of ${BUDGET_USD:.2f} spent (${spent:.2f})")
    usage = UsageMeter()
    started = time.monotonic()
    yield usage
    report = getattr(request.node, "matrix_report", None)
    record = {
        "model": model["id"],
        "test": request.node.originalname,
        "case": request.node.callspec.id,
        "outcome": report.outcome if report else "error",
        "llm_calls": usage.calls,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "cost_usd": round(usage.cost_usd, 6),
        "seconds": round(time.monotonic() - started, 2),
        "error": str(report.longrepr).splitlines()[-1][:300] if report and report.failed else None,
    }
    with RESULTS_FILE.open("a") as results:  # one short line per append, so workers never interleave
        results.write(json.dumps(record) + "\n")
