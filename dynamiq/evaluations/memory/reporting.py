"""Safe human and CI projections of recomputed memory evaluation results."""

import csv
import html
import io
import re
import xml.etree.ElementTree as ET  # nosec B405 - builds XML only; never parses input
from collections.abc import Iterator
from typing import Literal

from dynamiq.evaluations.memory.metrics import MemoryAssertion
from dynamiq.evaluations.memory.replay import replay_memory
from dynamiq.evaluations.memory.results import MemoryEvaluationReport
from dynamiq.evaluations.memory.serialization import dump_report
from dynamiq.evaluations.memory.suite import MemoryEvaluationLimits, Verdict

MemoryReportFormat = Literal["json", "markdown", "csv", "junit"]


def _visible(text: str) -> str:
    """Keep untrusted logical labels from emitting terminal or bidi controls."""
    return "".join(
        (
            f"\\u{ord(character):04x}"
            if ord(character) < 32
            or 127 <= ord(character) <= 159
            or 0x202A <= ord(character) <= 0x202E
            or 0x2066 <= ord(character) <= 0x2069
            else character
        )
        for character in text
    )


def _markdown(text: str) -> str:
    escaped = re.sub(r"([\\`*_{}\[\]()#+.!>~\-])", r"\\\1", _visible(text))
    return html.escape(escaped, quote=True).replace("|", "&#124;")


def _csv_cell(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    # CSV quoting is not a formula-injection defense. Prefix the cell before
    # passing it to csv.writer, including whitespace-obscured formula markers.
    if text.lstrip().startswith(("=", "+", "-", "@")) or text.startswith(("\t", "\r", "\n")):
        text = "'" + text
    return _visible(text)


def _number(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.6g}"


def _details(report: MemoryEvaluationReport) -> Iterator[tuple[str, str, MemoryAssertion]]:
    for trial in report.trials:
        scope = f"{trial.case_id} / repetition {trial.repetition + 1}"
        for operation in trial.operations:
            for check in operation.assertions:
                yield scope, operation.step_id, check
        for check in trial.cleanup:
            yield scope, "cleanup", check
    for gate in report.gates:
        scope = f"{gate.gate.group} / {gate.gate.metric.value}@{gate.gate.cutoff}"
        for check in gate.assertions:
            yield scope, "gate", check


def _markdown_report(report: MemoryEvaluationReport, maximum: int) -> str:
    lines = [
        f"# Memory evaluation: {_markdown(report.evidence.specification.name)}",
        "",
        f"Verdict: **{report.verdict.value}**",
        f"Completed operations: {report.completed_operations}/{report.expected_operations}.",
        "",
        "Observed means weight cases equally, then repetitions, then queries. "
        "They describe defined observed scores; coverage and support remain separate.",
        "",
        "| Group | Metric | k | Value | Available / expected queries | Defined support | Coverage |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report.aggregates:
        lines.append(
            f"| {_markdown(row.group)} | {row.metric.value} | {row.cutoff} | {_number(row.value)} | "
            f"{row.available_queries}/{row.expected_queries} | {row.supported_queries} | {_number(row.coverage)} |"
        )
    lines.extend(
        (
            "",
            "## Requirements",
            "",
            "| Case / group | Step | Requirement | Verdict | Reason |",
            "| --- | --- | --- | --- | --- |",
        )
    )
    total = 0
    for scope, step, check in _details(report):
        total += 1
        if total <= maximum:
            lines.append(
                f"| {_markdown(scope)} | {_markdown(step)} | {_markdown(check.name)} | "
                f"{check.verdict.value} | {_markdown(check.reason)} |"
            )
    if total > maximum:
        lines.extend(
            (
                "",
                f"Showing {maximum} of {total} requirement rows. The verdict uses all requirements; "
                "the JSON evidence retains the complete replay ledger.",
            )
        )
    lines.extend(
        (
            "",
            "Evidence excludes captured memory/query text, provider IDs and exception messages. "
            "Caller-supplied logical labels remain visible. Native backend logs are outside this export.",
            "",
        )
    )
    return "\n".join(lines)


def _csv_report(report: MemoryEvaluationReport) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        (
            "suite",
            "run_id",
            "verdict",
            "group",
            "metric",
            "cutoff",
            "value",
            "expected_queries",
            "available_queries",
            "supported_queries",
            "undefined_queries",
            "coverage",
            "expected_trials",
            "supported_trials",
            "expected_cases",
            "supported_cases",
            "reason",
        )
    )
    for row in report.aggregates:
        writer.writerow(
            _csv_cell(value)
            for value in (
                report.evidence.specification.name,
                report.evidence.run_id,
                report.verdict.value,
                row.group,
                row.metric.value,
                row.cutoff,
                row.value,
                row.expected_queries,
                row.available_queries,
                row.supported_queries,
                row.undefined_queries,
                row.coverage,
                row.expected_trials,
                row.supported_trials,
                row.expected_cases,
                row.supported_cases,
                row.reason,
            )
        )
    return output.getvalue()


def _xml_text(text: str) -> str:
    # XML 1.0 cannot represent these code points, even as numeric entities.
    return "".join(
        (
            character
            if character in "\t\n\r"
            or 32 <= ord(character) <= 0xD7FF
            or 0xE000 <= ord(character) <= 0xFFFD
            or 0x10000 <= ord(character) <= 0x10FFFF
            else "\ufffd"
        )
        for character in _visible(text)
    )


def _junit_report(report: MemoryEvaluationReport, maximum: int) -> str:
    root = ET.Element("testsuite", name=_xml_text(report.evidence.specification.name))
    failures = skipped = 0

    def add(scope: str, name: str, verdict: Verdict, reason: str) -> None:
        nonlocal failures, skipped
        case = ET.SubElement(root, "testcase", classname=_xml_text(scope), name=_xml_text(name))
        if verdict == Verdict.FAIL:
            failures += 1
            ET.SubElement(case, "failure", message=_xml_text(reason)).text = _xml_text(reason)
        elif verdict in (Verdict.INCONCLUSIVE, Verdict.NOT_EVALUATED):
            skipped += 1
            ET.SubElement(case, "skipped", message=_xml_text(reason))

    # The global decision is always present, even when detail output is capped.
    add("memory-evaluation", "overall", report.verdict, "complete_ledger_verdict")
    total = 0
    for scope, step, check in _details(report):
        total += 1
        if total <= maximum:
            add(scope, f"{step} / {check.name}", check.verdict, check.reason)
    if total > maximum:
        add(
            "memory-evaluation",
            "detail-output",
            Verdict.NOT_EVALUATED,
            f"presentation truncated: {maximum} of {total} requirement rows",
        )
    root.set("tests", str(len(root)))
    root.set("failures", str(failures))
    root.set("errors", "0")
    root.set("skipped", str(skipped))
    return ET.tostring(root, encoding="unicode", xml_declaration=True) + "\n"


def render_memory_report(
    report: MemoryEvaluationReport,
    *,
    format: MemoryReportFormat = "markdown",
    limits: MemoryEvaluationLimits | None = None,
) -> str:
    """Render recomputed results; presentation truncation never changes verdicts.

    JSON includes full evidence and every result. CSV contains aggregate rows.
    Markdown and JUnit cap only detailed requirement rows. A CI caller should
    inspect the explicit verdict; JUnit marks inconclusive evidence as skipped.
    """
    trusted = MemoryEvaluationLimits.model_validate((limits or MemoryEvaluationLimits()).model_dump())
    checked = replay_memory(report.evidence, limits=trusted)
    if format == "json":
        return dump_report(checked, limits=trusted)
    if format == "markdown":
        text = _markdown_report(checked, trusted.max_detail_rows)
    elif format == "csv":
        text = _csv_report(checked)
    elif format == "junit":
        text = _junit_report(checked, trusted.max_detail_rows)
    else:
        raise ValueError("unsupported memory report format")
    if len(text.encode("utf-8")) > trusted.max_bytes:
        raise ValueError("memory report byte limit exceeded")
    return text
