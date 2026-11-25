# Evaluate long-term memory lifecycles

`dynamiq.evaluations.memory` evaluates the existing `LongTermMemoryBackend`
interface across ordered writes, recalls, updates and deletes. It captures
selected observations for offline replay and regression comparison. It does
not replace a memory backend or use an LLM judge.

## Run a scenario

Supply an explicit backend factory. A new generated user scope is allocated
for each logical actor in each case/repetition. The factory is called once per
trial and may return a shared backend; generated scopes remain disjoint.

```python
from dynamiq.evaluations.memory import (
    Forget, MemoryCase, MemoryGate, MemorySuite, Recall, Remember,
    evaluate_memory, render_memory_report, write_evidence,
)

suite = MemorySuite(
    name="preference-lifecycle",
    dataset_revision="preferences-v1",  # nonsecret input revision asserted by you
    cutoffs=(1,),
    repetitions=3,
    cases=(MemoryCase(id="preference", steps=(
        Remember(id="write", actor="subject", alias="preference", content="Prefers tea"),
        Recall(id="find", actor="subject", query="Preferred drink", limit=1,
               relevance=({"alias": "preference", "grade": 3},)),
        Forget(id="delete", actor="subject", alias="preference"),
        Recall(id="verify-delete", actor="subject", query="Preferred drink", limit=1,
               expect_empty=True, forbidden=("preference",)),
    )),),
    gates=(MemoryGate(metric="forbidden_query_rate", cutoff=1, maximum=0.0),),
)

# backend_factory is supplied by your application, not loaded from a file.
report = evaluate_memory(suite, backend_factory)
write_evidence("memory-evidence.json", report.evidence)
markdown = render_memory_report(report)
```

The example at `examples/evaluations/memory_lifecycle.py` runs without provider
credentials using the actual in-memory backend and a deliberately constant
local embedding fixture. It demonstrates lifecycle behavior, not semantic
retrieval quality. An application-supplied embedder/backend may make network
calls and incur provider costs during explicit live evaluation. Parsing,
replay, comparison and reporting do not call a provider.

Operations execute sequentially and are never retried. An operation exception
stops that trial; its remaining expected steps remain unavailable in the
report. Later trials still run. Configure timeouts on the supplied backend:
the evaluator cannot safely cancel a synchronous in-flight write.

By default, every generated scope is cleared in `finally`, followed by a
bounded `list_all(..., limit=1)` emptiness check. All actors are attempted even
when another cleanup fails. Cleanup errors prevent a pass. Cancellation is
re-raised after cleanup and caller-owned clients are never closed.
`cleanup_enabled=False` explicitly skips cleanup; it is reported as not
evaluated. The harness counts its own backend method calls, not internal
embedding requests or database operations.

## Identity, updates and requirements

A `Remember` alias captures the returned object and its exact content version.
It does not follow a later mutation of a returned `Fact`. A semantic upsert may
reuse the object ID while creating a new version. `expected_same_as` can
require that reuse; `expected_outcome` can require `created`, `updated` or
`unchanged`. An `unchanged` result must match the current captured version.

Write integrity always checks the returned text against the requested text.
The default `write_match="native"` compares stripped, lowercased strings to
support native case-insensitive deduplication. `write_match="exact"` requires
stripped text equality. A backend returning unrelated text fails this check,
even if that text is subsequently retrievable.

`Forget` addresses the current object at an alias's ID, so deleting through an
older alias tombstones all captured versions of that object. A cross-actor
delete control can reference an earlier alias from another test actor; caller
input cannot specify real backend IDs or user scopes. Returning another
actor's data or deleting its object is always an isolation failure.

Recall judgments may reference only earlier aliases. Equivalent aliases
coalesce after native deduplication; conflicting grades are rejected. Two
different versions of the same object cannot both be positive targets in one
query because duplicate object IDs cannot earn relevance twice. A forbidden
alias can refer to an earlier version or another actor's test data.

Invalid/foreign rows retain their original rank and cannot earn relevance.
Duplicate object IDs occupy ranks but earn credit only once. Unknown IDs or
unknown content versions remain explicit. Extra results beyond the requested
limit cause a backend-output violation. Empty-result expectations inspect the
entire bounded returned list, not only the smallest metric cutoff.

## Metrics and coverage

Each configured cutoff is reported independently:

| Metric | Definition |
| --- | --- |
| `precision` | Relevant unique versions divided by k, even if fewer than k hits are returned |
| `recall` | Relevant unique versions found divided by positive judged versions |
| `reciprocal_rank` | Reciprocal original rank of the first relevant unique version |
| `ndcg` | DCG divided by ideal DCG, using `(2**grade - 1) / log2(rank + 1)` |
| `hit_rate` | Whether any relevant unique version was returned |
| `forbidden_query_rate` | Whether the query returned any forbidden version within k |
| `duplicate_query_rate` | Whether the query returned a duplicate object ID within k |
| `unknown_query_rate` | Whether an otherwise valid in-scope hit had no captured version within k |
| `scope_violation_query_rate` | Whether the query returned another actor's data within k |

The adverse rates are binary query indicators, not fractions of hit rows.
With no positive judgments, recall, reciprocal rank, NDCG and hit rate are
undefined. They are not assigned perfect scores. Use `expect_empty` for a
negative query and distinguish structural undefinedness from missing execution.

Aggregation averages queries within a repetition, repetitions within a case,
then cases equally. Long cases cannot dominate short cases. Means describe
available, defined observations. Expected/available/supported counts and
coverage retain failed factories, failed operations and absent trial records.

`MemoryGate` accepts a minimum and/or maximum, plus minimum defined support
and minimum availability coverage (default 1.0). Bounds apply to observed
means, with no population inference. A known violated bound wins over
incomplete evidence. Lifecycle invariants are checked independently of metric
gates. A passing lifecycle does not establish retrieval quality beyond the
requirements that you configured.

## Compare a backend or configuration change

```python
from dynamiq.evaluations.memory import (
    MemoryComparisonGate, compare_memory, read_evidence, render_memory_comparison,
)

comparison = compare_memory(
    read_evidence("baseline.json"),
    read_evidence("candidate.json"),
    gates=(MemoryComparisonGate(
        metric="recall", cutoff=1, minimum_improvement=-0.02,
        minimum_support=10, minimum_coverage=1.0,
    ),),
)
markdown = render_memory_comparison(comparison)
```

Comparison requires matching execution/reference specifications and a present,
equal `dataset_revision`. The revision is your assertion that the raw scenario
inputs match; it is not a content hash or authenticity proof. Repetition
indices identify corresponding planned slots, not shared random draws.

Both paired means use exactly the same defined, comparable query slots.
Changing runtime alias coalescing can change a metric's denominator, so the
comparison checks logical alias object/version equivalence partitions. A
changed partition makes that query incomparable and reduces paired coverage.
Run-local opaque identity strings are never compared across runs.

Raw delta is candidate minus baseline. Oriented improvement reverses the sign
for adverse query rates, so positive always means better. A negative
`minimum_improvement` permits bounded degradation. Candidate lifecycle or
cleanup failures block acceptance even when retrieval improves. Source
absolute metric gate verdicts are reported separately; they are not silently
reused as comparison gates. Without explicit comparison gates, complete
observations are descriptive and `not_evaluated`, not an automatic pass.

## Portable input and offline CLI

```sh
dynamiq evaluation memory validate scenario.json
dynamiq evaluation memory replay memory-evidence.json --format markdown --out report.md
dynamiq evaluation memory replay memory-evidence.json --format junit --out results.xml
dynamiq evaluation memory compare baseline.json candidate.json \
  --metric recall --cutoff 1 --minimum-improvement -0.02 --format json --out comparison.json
```

No API credentials or dynamic backend imports are needed. Replay/compare write
the report before returning the verdict exit code: 0 pass, 1 fail, 2
inconclusive, 3 not evaluated. `--no-check` returns success after valid report
generation regardless of verdict. Invalid input is a CLI error. Existing
output files require `--overwrite`; publication is atomic.

Use `dump_suite`/`write_suite` only when you explicitly want to export raw
memory/query text. `dump_evidence`/`write_evidence` export selected observations
and a text-free expected ledger. Replay rejects duplicate/out-of-order steps,
impossible identity links and inconsistent call counts. Missing expected slots
are reconstructed as unavailable. Replay does not establish the authenticity
of a supplied capture.

Strict JSON rejects duplicate keys, nonfinite numbers, invalid Unicode,
unsupported versions and unknown fields. `MemoryEvaluationLimits` provides
trusted byte/depth/node, trial/operation/cleanup, hit and metric-cell budgets;
an input file cannot raise them. Configure these limits before live evaluation.

Reports exclude captured memory/query text, metadata, native IDs, generated
scopes, raw content hashes and provider exception messages. **Logical labels
and dataset revisions remain visible**; choose nonsecret values. Native
backend logging is outside this export's privacy boundary.

JSON reports retain complete evidence. Use `load_report`/`read_report` to restore a saved
JSON report; all derived values are recomputed from its embedded evidence.
`load_evidence`/`read_evidence` accept the smaller capture-only artifact. CSV contains
aggregate rows and escapes
spreadsheet formula markers. Markdown/JUnit cap detailed requirement rows
without changing the complete-ledger verdict. JUnit marks inconclusive results
as skipped and always includes the global verdict; use the CLI exit code when
CI must reject incomplete evidence.
