import hashlib

from dynamiq.connections import BaseConnection
from dynamiq.evaluations.memory.replay import replay_memory
from dynamiq.evaluations.memory.runner import evaluate_memory
from dynamiq.evaluations.memory.suite import Forget, MemoryCase, MemorySuite, Recall, Remember, Verdict
from dynamiq.memory.long_term.backends.in_memory import InMemoryLongTermMemoryBackend
from dynamiq.nodes.embedders.base import TextEmbedder, TextEmbedderInputSchema


class LocalConnection(BaseConnection):
    def connect(self):
        return None


class LocalEmbedder(TextEmbedder):
    connection: BaseConnection = LocalConnection()

    def execute(self, input_data: TextEmbedderInputSchema, config=None, **kwargs):
        text = input_data.query
        digest = hashlib.sha256(text.encode()).digest()
        return {"query": text, "embedding": [value / 255 for value in digest[:8]]}


def test_native_update_dedup_foreign_delete_forget_and_replay():
    backend = InMemoryLongTermMemoryBackend(embedder=LocalEmbedder(), upsert_threshold=0.0)
    scenario = MemorySuite(
        name="native",
        cutoffs=(1,),
        repetitions=2,
        cases=(
            MemoryCase(
                id="lifecycle",
                actors=("subject", "other"),
                steps=(
                    Remember(
                        id="create", actor="subject", alias="old", content="Likes tea", expected_outcome="created"
                    ),
                    Remember(
                        id="update",
                        actor="subject",
                        alias="new",
                        content="Likes coffee",
                        expected_outcome="updated",
                        expected_same_as="old",
                    ),
                    Remember(
                        id="dedup",
                        actor="subject",
                        alias="same",
                        content="Likes coffee",
                        expected_outcome="unchanged",
                        expected_same_as="new",
                    ),
                    Recall(
                        id="updated",
                        actor="subject",
                        query="Likes coffee",
                        limit=1,
                        relevance=({"alias": "new", "grade": 3}, {"alias": "same", "grade": 3}),
                        forbidden=("old",),
                    ),
                    Forget(id="foreign-delete", actor="other", alias="new", expected_status="forbidden"),
                    Recall(
                        id="isolation",
                        actor="other",
                        query="Likes coffee",
                        limit=1,
                        expect_empty=True,
                        forbidden=("new",),
                    ),
                    Forget(id="delete", actor="subject", alias="old", expected_status="deleted"),
                    Recall(
                        id="absent",
                        actor="subject",
                        query="Likes coffee",
                        limit=1,
                        expect_empty=True,
                        forbidden=("new",),
                    ),
                ),
            ),
        ),
    )
    report = evaluate_memory(scenario, lambda: backend)
    assert report.verdict == Verdict.PASS
    assert report.completed_operations == 16
    assert replay_memory(report.evidence) == report
    assert "Likes coffee" not in report.model_dump_json()
