"""JSON and Markdown projections of paired memory regression evidence."""

from typing import Literal

from dynamiq.evaluations.memory.comparison import MemoryComparisonReport, compare_memory
from dynamiq.evaluations.memory.reporting import _markdown, _number
from dynamiq.evaluations.memory.serialization import _encode
from dynamiq.evaluations.memory.suite import MemoryEvaluationLimits


def render_memory_comparison(
    report: MemoryComparisonReport,
    *,
    format: Literal["json", "markdown"] = "markdown",
    limits: MemoryEvaluationLimits | None = None,
) -> str:
    """Recompute before rendering; retain both complete evidence inputs in JSON."""
    trusted = MemoryEvaluationLimits.model_validate((limits or MemoryEvaluationLimits()).model_dump())
    checked = compare_memory(
        report.baseline, report.candidate, gates=tuple(result.gate for result in report.gates), limits=trusted
    )
    if format == "json":
        text = _encode(checked, trusted)
    elif format == "markdown":
        lines = [
            "# Memory regression comparison",
            "",
            f"Verdict: **{checked.verdict.value}**",
            f"Scenario revision asserted by caller: {_markdown(checked.dataset_revision)}.",
            f"Baseline run: {_markdown(checked.baseline.run_id)}. "
            f"Candidate run: {_markdown(checked.candidate.run_id)}.",
            "",
            "Both means use identical comparable query pairs, weighted equally by case, then repetition, then query. "
            "Delta is candidate minus baseline. Improvement reverses the sign for adverse query rates, "
            "so positive always means better. These are descriptive observations, not significance tests.",
            "",
            f"Candidate lifecycle/cleanup verdict: **{checked.candidate_lifecycle_verdict.value}**. "
            f"Source absolute verdicts: baseline **{checked.baseline_absolute_verdict.value}**, "
            f"candidate **{checked.candidate_absolute_verdict.value}**. "
            "Source absolute metric gates are reported separately and are not comparison requirements.",
            "",
            "| Group | Metric | k | Baseline | Candidate | Delta | Improvement | Paired / expected | Coverage |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in checked.metrics:
            lines.append(
                f"| {_markdown(row.group)} | {row.metric.value} | {row.cutoff} | "
                f"{_number(row.baseline_mean)} | {_number(row.candidate_mean)} | {_number(row.delta)} | "
                f"{_number(row.improvement)} | {row.paired_queries}/{row.expected_queries} | {_number(row.coverage)} |"
            )
        lines.extend(
            (
                "",
                "## Pair availability",
                "",
                "| Group | Metric | k | Baseline defined | Candidate defined | "
                "Unavailable | Undefined | Incomparable |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            )
        )
        for row in checked.metrics:
            lines.append(
                f"| {_markdown(row.group)} | {row.metric.value} | {row.cutoff} | "
                f"{row.baseline_defined_queries} | {row.candidate_defined_queries} | {row.unavailable_queries} | "
                f"{row.undefined_queries} | {row.incomparable_queries} |"
            )
        lines.extend(
            (
                "",
                "A changed alias object/version partition makes the query incomparable. "
                "Unavailable and undefined slots stay in expected coverage.",
                "",
                "## Comparison requirements",
                "",
            )
        )
        if not checked.gates:
            lines.append("No comparison gates were configured; observed differences alone do not establish acceptance.")
        else:
            lines.extend(
                (
                    "| Group | Metric | k | Minimum improvement | Observed | Verdict |",
                    "| --- | --- | ---: | ---: | ---: | --- |",
                )
            )
            for result in checked.gates:
                gate = result.gate
                lines.append(
                    f"| {_markdown(gate.group)} | {gate.metric.value} | {gate.cutoff} | "
                    f"{_number(gate.minimum_improvement)} | {_number(result.improvement)} | {result.verdict.value} |"
                )
        lines.extend(
            (
                "",
                "The caller asserts that the revision identifies the same raw scenario inputs. "
                "This does not verify model configuration or randomness. "
                "Logical labels remain visible; captured text and provider identifiers are excluded.",
                "",
            )
        )
        text = "\n".join(lines)
    else:
        raise ValueError("unsupported memory comparison report format")
    if len(text.encode("utf-8")) > trusted.max_bytes:
        raise ValueError("memory comparison report byte limit exceeded")
    return text
