from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from dynamiq.memory.long_term import LongTermMemoryBackend, RememberOutcome
from dynamiq.nodes.node import Node, ensure_config
from dynamiq.nodes.types import NodeGroup
from dynamiq.runnables import RunnableConfig
from dynamiq.types.cancellation import check_cancellation
from dynamiq.utils.logger import logger

REMEMBER_DESCRIPTION = """Persist a durable fact about the current user to long-term memory.

Key capabilities:
- Survives across conversations and sessions — not just this chat
- Idempotent on identical content (re-stating the same fact is a no-op)
- Semantic upsert: re-stating a near-paraphrase REPLACES the older version in place,
  which is how you correct or update what you previously remembered
- Optional structured metadata (e.g. category, source) for later filtering

Usage strategy:
- Call when the user explicitly says "remember…", "save this…", "keep in mind for next time…"
- Call when you have learned something that will clearly matter in future sessions
  (a stable preference, a constraint, a recurring context, biographical info)
- To CORRECT a previously-saved fact, just call this tool again with the corrected
  statement — the older paraphrase is replaced automatically.
- Do NOT use for ephemeral turn-level state — that is what the conversation history is for
- Do NOT use to remember tool outputs, file paths, or anything tied to this run
- The fact is scoped to the current user automatically; never pass or invent a user id

Returns: a short status line — "Fact saved.", "Fact updated.", or "Already remembered."

Examples:
- {"content": "Prefers dogs over cats"}
- {"content": "Allergic to peanuts", "metadata": {"category": "health"}}
- {"content": "Works in EST timezone", "metadata": {"category": "context"}}
"""

RECALL_DESCRIPTION = """Search the user's long-term memory for facts relevant to one or more queries.

Key capabilities:
- Semantic search (not keyword) — matches meaning, paraphrases, synonyms
- Scoped to the current user automatically — never crosses users
- Multi-query: pass several phrasings in one call; results are merged and de-duplicated.
  Because matches are sensitive to phrasing, supplying 2–4 angles per recall typically
  improves recall over a single query without an extra round-trip.
- Returns the most relevant facts first, as plain text

Usage strategy:
- Call PROACTIVELY at the start of a turn when the request hints at something personal
  (preferences, past decisions, biographical info, recurring context)
- Call BEFORE answering questions where prior context would change your response
- Prefer 2–4 distinct phrasings over a single query when the topic is fuzzy
- If no relevant facts are found, just proceed without them
- Skip when the question is purely factual or has no user-specific component

Returns: a bullet list of relevant facts (most relevant first), or "No relevant facts."

Examples:
- {"queries": ["food preferences"]}
- {"queries": ["what does the user do for work?", "user profession", "user job role"], "limit": 3}
- {"queries": ["timezone", "working hours", "schedule constraints"], "limit": 10}
"""


class RememberFactInputSchema(BaseModel):
    """LLM-visible input for `remember_fact`. `user_id` is bound at construction."""

    content: str = Field(..., min_length=1, max_length=1000,
                         description="The fact to remember, as a short statement.")
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional free-form metadata (e.g. {'category': 'preference'}).",
    )


class RecallFactsInputSchema(BaseModel):
    """LLM-visible input for `recall_facts`. `user_id` is bound at construction."""

    queries: list[str] = Field(
        ...,
        min_length=1,
        max_length=5,
        description=(
            "One or more search phrasings. Semantic search is phrasing-sensitive, "
            "so 2–4 distinct angles usually beat a single query for fuzzy topics."
        ),
    )
    limit: int = Field(default=5, ge=1, le=20,
                       description="Max facts to return after merging across queries.")

    @field_validator("queries", mode="after")
    @classmethod
    def _strip_and_require_nonblank(cls, queries: list[str]) -> list[str]:
        """Reject whitespace-only entries here so the model sees a clean
        validation error, instead of the backend raising at recall time."""
        cleaned = [q.strip() for q in queries]
        if any(not q for q in cleaned):
            raise ValueError("`queries` must not contain empty or whitespace-only strings")
        return cleaned


class _LongTermMemoryTool(Node):
    """Shared base for the long-term memory tools."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    backend: LongTermMemoryBackend
    user_id: str

    @property
    def to_dict_exclude_params(self) -> dict[str, Any]:
        return super().to_dict_exclude_params | {"backend": True}

    def to_dict(self, include_secure_params: bool = False, **kwargs) -> dict[str, Any]:
        data = super().to_dict(include_secure_params=include_secure_params, **kwargs)
        data["backend"] = self.backend.to_dict(include_secure_params=include_secure_params, **kwargs)
        return data


_OUTCOME_MESSAGES: dict[RememberOutcome, str] = {
    RememberOutcome.CREATED: "Fact saved.",
    RememberOutcome.UPDATED: "Fact updated.",
    RememberOutcome.UNCHANGED: "Already remembered.",
}


class RememberFactTool(_LongTermMemoryTool):
    """Write a fact to long-term memory, scoped to the bound user_id."""

    name: str = "remember_fact"
    description: str = REMEMBER_DESCRIPTION
    input_schema: ClassVar[type[RememberFactInputSchema]] = RememberFactInputSchema

    def execute(
        self, input_data: RememberFactInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        logger.debug(f"Tool {self.name} - {self.id}: started")
        config = ensure_config(config)
        check_cancellation(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        fact, outcome = self.backend.remember(
            content=input_data.content,
            user_id=self.user_id,
            metadata=input_data.metadata,
        )
        if self.is_optimized_for_agents:
            return {"content": _OUTCOME_MESSAGES[outcome]}
        return {"content": {"fact_id": fact.id, "outcome": outcome.value}}


class RecallFactsTool(_LongTermMemoryTool):
    """Search long-term memory for facts relevant to a query, scoped to user_id."""

    name: str = "recall_facts"
    description: str = RECALL_DESCRIPTION
    input_schema: ClassVar[type[RecallFactsInputSchema]] = RecallFactsInputSchema

    def execute(
        self, input_data: RecallFactsInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        logger.debug(f"Tool {self.name} - {self.id}: started")
        config = ensure_config(config)
        check_cancellation(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        # Recall once per query; merge by fact id keeping the best score so a
        # paraphrase that scores higher under one phrasing isn't penalised by
        # another phrasing's weaker hit. Ask each backend call for `limit` to
        # let the union be re-ranked at the end.
        best: dict[str, tuple[Any, float]] = {}
        for query in input_data.queries:
            for fact, score in self.backend.recall(query=query, user_id=self.user_id, limit=input_data.limit):
                prev = best.get(fact.id)
                if prev is None or score > prev[1]:
                    best[fact.id] = (fact, score)

        hits = sorted(best.values(), key=lambda pair: pair[1], reverse=True)[: input_data.limit]

        if self.is_optimized_for_agents:
            if not hits:
                return {"content": "No relevant facts."}
            return {"content": "\n".join(f"- {fact.content}" for fact, _ in hits)}
        return {"content": [{"content": fact.content, "score": round(score, 4)} for fact, score in hits]}


def build_long_term_memory_tools(*, backend: LongTermMemoryBackend, user_id: str) -> list[Node]:
    """Construct the long-term-memory tools (remember + recall) with `user_id` baked in.

    Per-tool subsetting was intentionally removed. If you need to expose only
    one of the two tools to a particular agent, instantiate the class directly
    instead of going through this factory. See the LTM plan v2 "Reversible
    cuts" appendix for the prior selector design.
    """
    return [
        RememberFactTool(backend=backend, user_id=user_id),
        RecallFactsTool(backend=backend, user_id=user_id),
    ]
