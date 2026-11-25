"""Credential-free validation and replay of portable memory evaluations."""

import json
from pathlib import Path

import click

from dynamiq.evaluations.memory.comparison import MemoryComparisonGate, compare_memory
from dynamiq.evaluations.memory.comparison_reporting import render_memory_comparison
from dynamiq.evaluations.memory.replay import replay_memory
from dynamiq.evaluations.memory.reporting import render_memory_report
from dynamiq.evaluations.memory.serialization import read_evidence, read_suite, write_text_atomic
from dynamiq.evaluations.memory.suite import MemoryEvaluationLimits, MemoryMetric, Verdict

memory_evaluation = click.Group(
    name="memory",
    help="Validate memory scenarios and replay captured evidence locally, without API credentials.",
)


def _limits(max_bytes: int, max_operations: int, max_details: int = 10000) -> MemoryEvaluationLimits:
    return MemoryEvaluationLimits(max_bytes=max_bytes, max_operations=max_operations, max_detail_rows=max_details)


def input_limits(function):
    function = click.option(
        "--max-bytes",
        type=click.IntRange(1, 64 * 1024 * 1024),
        default=64 * 1024 * 1024,
        show_default=True,
        help="Trusted input/output byte ceiling; files cannot raise it.",
    )(function)
    return click.option(
        "--max-operations",
        type=click.IntRange(1, 100000),
        default=100000,
        show_default=True,
        help="Maximum expected operation slots, including unavailable operations.",
    )(function)


@memory_evaluation.command("validate")
@click.argument("suite_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@input_limits
def validate_memory_suite(suite_path: Path, max_bytes: int, max_operations: int) -> None:
    """Check a raw-text scenario without running its memory operations.

    Output contains structural counts only. The suite file itself includes
    memory/query text and should be handled as application input data.
    """
    try:
        suite = read_suite(suite_path, limits=_limits(max_bytes, max_operations))
    except (OSError, ValueError) as exc:
        raise click.ClickException(f"Memory suite validation failed: {exc}") from None
    click.echo(
        json.dumps(
            {
                "valid": True,
                "cases": len(suite.cases),
                "repetitions": suite.repetitions,
                "expected_trials": len(suite.cases) * suite.repetitions,
                "expected_operations": sum(len(case.steps) for case in suite.cases) * suite.repetitions,
                "cutoffs": suite.cutoffs,
            },
            allow_nan=False,
        )
    )


@memory_evaluation.command("replay")
@click.argument("evidence_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--format", "report_format", type=click.Choice(("markdown", "json", "csv", "junit")), default="markdown")
@click.option("--out", "out_path", type=click.Path(dir_okay=False, path_type=Path), default=None)
@click.option("--overwrite", is_flag=True, help="Atomically replace an existing output file.")
@click.option("--check/--no-check", default=True, help="Exit 1 for fail, 2 for inconclusive, 3 for not evaluated.")
@click.option(
    "--max-details",
    type=click.IntRange(1, 1000000),
    default=10000,
    help="Maximum Markdown/JUnit detail rows; the verdict still uses all evidence.",
)
@input_limits
def replay_memory_evidence(
    evidence_path: Path,
    report_format: str,
    out_path: Path | None,
    overwrite: bool,
    check: bool,
    max_details: int,
    max_bytes: int,
    max_operations: int,
) -> None:
    """Recompute a report offline from selected observations, without provider calls.

    Reports are written before the verdict exit code is returned. JSON includes
    the complete evidence; other formats are presentation projections. Evidence
    is not an authenticity proof and logical labels remain visible in reports.
    """
    trusted = _limits(max_bytes, max_operations, max_details)
    try:
        evidence = read_evidence(evidence_path, limits=trusted)
        report = replay_memory(evidence, limits=trusted)
        text = render_memory_report(report, format=report_format, limits=trusted)
        if out_path is None:
            click.echo(text, nl=False)
        else:
            write_text_atomic(out_path, text, overwrite=overwrite)
    except (OSError, ValueError) as exc:
        raise click.ClickException(f"Memory evidence replay failed: {exc}") from None
    if check:
        exit_code = {Verdict.PASS: 0, Verdict.FAIL: 1, Verdict.INCONCLUSIVE: 2, Verdict.NOT_EVALUATED: 3}[
            report.verdict
        ]
        if exit_code:
            raise click.exceptions.Exit(exit_code)


@memory_evaluation.command("compare")
@click.argument("baseline_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("candidate_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--format", "report_format", type=click.Choice(("markdown", "json")), default="markdown")
@click.option("--out", "out_path", type=click.Path(dir_okay=False, path_type=Path), default=None)
@click.option("--overwrite", is_flag=True)
@click.option(
    "--metric",
    type=click.Choice(tuple(metric.value for metric in MemoryMetric)),
    default=None,
    help="Configure one explicit regression gate; omit for a descriptive comparison.",
)
@click.option("--cutoff", type=click.IntRange(1, 1000), default=None)
@click.option("--group", default="overall", show_default=True)
@click.option(
    "--minimum-improvement",
    type=click.FloatRange(-1.0, 1.0),
    default=None,
    help="Positive means better; negative explicitly permits bounded degradation.",
)
@click.option("--minimum-support", type=click.IntRange(1, 1000000), default=1)
@click.option("--minimum-coverage", type=click.FloatRange(0.0, 1.0), default=1.0)
@click.option("--check/--no-check", default=True)
@input_limits
def compare_memory_evidence(
    baseline_path: Path,
    candidate_path: Path,
    report_format: str,
    out_path: Path | None,
    overwrite: bool,
    metric: str | None,
    cutoff: int | None,
    group: str,
    minimum_improvement: float | None,
    minimum_support: int,
    minimum_coverage: float,
    check: bool,
    max_bytes: int,
    max_operations: int,
) -> None:
    """Compare two captured runs on identical comparable query pairs.

    Both files need the same explicit dataset_revision and compatible scenario
    specifications. The revision is the caller's assertion about raw inputs.
    Omit --metric for descriptive output (exit 3 unless --no-check). A single
    CLI gate requires --metric, --cutoff and --minimum-improvement together;
    Python callers can provide multiple typed gates.
    """
    supplied = (metric is not None, cutoff is not None, minimum_improvement is not None)
    if any(supplied) and not all(supplied):
        raise click.UsageError("--metric, --cutoff and --minimum-improvement must be supplied together")
    trusted = _limits(max_bytes, max_operations)
    try:
        gates = (
            ()
            if metric is None
            else (
                MemoryComparisonGate(
                    metric=metric,
                    cutoff=cutoff,
                    group=group,
                    minimum_improvement=minimum_improvement,
                    minimum_support=minimum_support,
                    minimum_coverage=minimum_coverage,
                ),
            )
        )
        result = compare_memory(
            read_evidence(baseline_path, limits=trusted),
            read_evidence(candidate_path, limits=trusted),
            gates=gates,
            limits=trusted,
        )
        text = render_memory_comparison(result, format=report_format, limits=trusted)
        if out_path is None:
            click.echo(text, nl=False)
        else:
            write_text_atomic(out_path, text, overwrite=overwrite)
    except (OSError, ValueError) as exc:
        raise click.ClickException(f"Memory comparison failed: {exc}") from None
    if check:
        exit_code = {Verdict.PASS: 0, Verdict.FAIL: 1, Verdict.INCONCLUSIVE: 2, Verdict.NOT_EVALUATED: 3}[
            result.verdict
        ]
        if exit_code:
            raise click.exceptions.Exit(exit_code)
