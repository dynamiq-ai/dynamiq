from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.memory.long_term import LongTermMemory, MemoryToolKind, RememberOutcome
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

RECALL_DESCRIPTION = """Search the user's long-term memory for facts relevant to a query.

Key capabilities:
- Semantic search (not keyword) — matches meaning, paraphrases, synonyms
- Scoped to the current user automatically — never crosses users
- Returns the most relevant facts first, as plain text

Usage strategy:
- Call PROACTIVELY at the start of a turn when the request hints at something personal
  (preferences, past decisions, biographical info, recurring context)
- Call BEFORE answering questions where prior context would change your response
- If no relevant facts are found, just proceed without them — no need to retry with
  different phrasings unless the user's request makes the prior context essential
- Skip when the question is purely factual or has no user-specific component

Returns: a bullet list of relevant facts (most relevant first), or "No relevant facts."

Examples:
- {"query": "food preferences"}
- {"query": "what does the user do for work?", "limit": 3}
- {"query": "timezone or schedule constraints", "limit": 10}
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

    query: str = Field(..., min_length=1, max_length=500,
                       description="What to search for.")
    limit: int = Field(default=5, ge=1, le=20,
                       description="Max facts to return.")


class _LongTermMemoryTool(Node):
    """Shared base for the long-term memory tools."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    long_term_memory: LongTermMemory
    user_id: str


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

        fact, outcome = self.long_term_memory.remember(
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

        hits = self.long_term_memory.recall(
            query=input_data.query,
            user_id=self.user_id,
            limit=input_data.limit,
        )
        if self.is_optimized_for_agents:
            if not hits:
                return {"content": "No relevant facts."}
            return {"content": "\n".join(f"- {fact.content}" for fact, _ in hits)}
        return {"content": [{"content": fact.content, "score": round(score, 4)} for fact, score in hits]}


_TOOL_BUILDERS: dict[MemoryToolKind, type[_LongTermMemoryTool]] = {
    MemoryToolKind.REMEMBER: RememberFactTool,
    MemoryToolKind.RECALL: RecallFactsTool,
}


def build_long_term_memory_tools(
    *,
    long_term_memory: LongTermMemory,
    user_id: str,
    include: tuple[MemoryToolKind | str, ...] = (
        MemoryToolKind.REMEMBER,
        MemoryToolKind.RECALL,
    ),
) -> list[Node]:
    """Construct long-term-memory tools with `user_id` baked in. Unknown keys in `include` are ignored."""
    tools: list[Node] = []
    for kind in include:
        try:
            tool_kind = MemoryToolKind(kind)
        except ValueError:
            continue
        tools.append(_TOOL_BUILDERS[tool_kind](long_term_memory=long_term_memory, user_id=user_id))
    return tools
