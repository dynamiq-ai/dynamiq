"""Run a deterministic native memory lifecycle without provider credentials."""

from pathlib import Path

import click

from dynamiq.connections import BaseConnection
from dynamiq.evaluations.memory import (
    Forget,
    MemoryCase,
    MemoryGate,
    MemorySuite,
    Recall,
    Remember,
    evaluate_memory,
    render_memory_report,
    write_evidence,
)
from dynamiq.memory.long_term.backends.in_memory import InMemoryLongTermMemoryBackend
from dynamiq.nodes.embedders.base import TextEmbedder, TextEmbedderInputSchema


class LocalConnection(BaseConnection):
    """The fixture needs no external client."""

    def connect(self) -> None:
        return None


class ConstantEmbedder(TextEmbedder):
    """Force the single fact per actor to upsert; this is not a quality model."""

    connection: BaseConnection = LocalConnection()

    def execute(self, input_data: TextEmbedderInputSchema, config=None, **kwargs) -> dict:
        return {"query": input_data.query, "embedding": [1.0, 0.0]}


def scenario() -> MemorySuite:
    return MemorySuite(
        name="local-memory-lifecycle",
        dataset_revision="local-lifecycle-v1",
        cutoffs=(1,),
        repetitions=2,
        cases=(
            MemoryCase(
                id="preference",
                actors=("subject", "other"),
                steps=(
                    Remember(
                        id="create", actor="subject", alias="old", content="Prefers tea", expected_outcome="created"
                    ),
                    Remember(
                        id="update",
                        actor="subject",
                        alias="current",
                        content="Prefers coffee",
                        expected_outcome="updated",
                        expected_same_as="old",
                    ),
                    Recall(
                        id="current-value",
                        actor="subject",
                        query="Preferred drink",
                        limit=1,
                        relevance=({"alias": "current", "grade": 3},),
                        forbidden=("old",),
                    ),
                    Forget(id="foreign-delete", actor="other", alias="current", expected_status="forbidden"),
                    Recall(
                        id="isolation",
                        actor="other",
                        query="Preferred drink",
                        limit=1,
                        expect_empty=True,
                        forbidden=("current",),
                    ),
                    Forget(id="delete-through-old-alias", actor="subject", alias="old"),
                    Recall(
                        id="verify-delete",
                        actor="subject",
                        query="Preferred drink",
                        limit=1,
                        expect_empty=True,
                        forbidden=("current",),
                    ),
                ),
            ),
        ),
        gates=(MemoryGate(metric="forbidden_query_rate", cutoff=1, maximum=0.0),),
    )


@click.command()
@click.option("--evidence", type=click.Path(dir_okay=False, path_type=Path), default=None)
@click.option("--overwrite", is_flag=True)
def main(evidence: Path | None, overwrite: bool) -> None:
    report = evaluate_memory(scenario(), lambda: InMemoryLongTermMemoryBackend(embedder=ConstantEmbedder()))
    if evidence is not None:
        write_evidence(evidence, report.evidence, overwrite=overwrite)
    click.echo(render_memory_report(report), nl=False)


if __name__ == "__main__":
    main()
