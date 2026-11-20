import csv
import io
import xml.etree.ElementTree as ET

import pytest

from dynamiq.evaluations.memory.evidence import MemoryEvidence, MemorySpecification
from dynamiq.evaluations.memory.replay import replay_memory
from dynamiq.evaluations.memory.reporting import render_memory_report
from dynamiq.evaluations.memory.suite import MemoryCase, MemoryEvaluationLimits, MemorySuite, Recall


def report():
    suite = MemorySuite(
        name='=HYPERLINK("https://example.invalid")',
        cutoffs=(1,),
        cases=(
            MemoryCase(
                id="![image](https://example.invalid)\x1b",
                steps=(Recall(id="q", actor="subject", query="secret", limit=1, expect_empty=True),),
            ),
        ),
    )
    return replay_memory(MemoryEvidence(run_id="run", specification=MemorySpecification.from_suite(suite)))


def test_markdown_neutralizes_embedded_links_and_terminal_controls():
    text = render_memory_report(report(), format="markdown")
    assert "![image](https://example.invalid)" not in text
    assert "\x1b" not in text
    assert "inconclusive" in text
    assert "secret" not in text


def test_csv_exports_undefined_as_empty_and_distinguishes_coverage():
    rows = list(csv.DictReader(io.StringIO(render_memory_report(report(), format="csv"))))
    assert rows[0]["suite"].startswith("'=HYPERLINK")
    assert rows[0]["value"] == ""
    assert rows[0]["coverage"] == "0.0"
    assert rows[0]["expected_queries"] == "1"


def test_junit_has_global_verdict_even_when_details_are_truncated():
    root = ET.fromstring(
        render_memory_report(report(), format="junit", limits=MemoryEvaluationLimits(max_detail_rows=1))
    )
    assert root.find("testcase/skipped") is not None
    assert root.attrib["tests"] == str(len(root.findall("testcase")))
    assert int(root.attrib["skipped"]) >= 1
    assert "\x1b" not in ET.tostring(root, encoding="unicode")


def test_json_recomputes_tampered_summary_and_keeps_complete_evidence():
    original = report()
    tampered = original.model_copy(update={"verdict": "pass", "completed_operations": 999})
    text = render_memory_report(tampered, format="json")
    assert '"verdict": "inconclusive"' in text
    assert '"completed_operations": 0' in text
    assert '"evidence"' in text


def test_unknown_format_rejected():
    with pytest.raises(ValueError):
        render_memory_report(report(), format="html")
