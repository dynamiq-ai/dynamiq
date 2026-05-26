from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.memory.long_term import LongTermMemory, MemoryToolKind
from dynamiq.nodes.node import Node, ensure_config
from dynamiq.nodes.types import NodeGroup
from dynamiq.runnables import RunnableConfig
from dynamiq.types.cancellation import check_cancellation
from dynamiq.utils.logger import logger

REMEMBER_DESCRIPTION = """Persist a durable fact about the current user to long-term memory.

Key capabilities:
- Survives across conversations and sessions — not just this chat
- Idempotent on identical content (same text returns the same fact_id, no duplicates)
- Optional structured metadata (e.g. category, source) for later filtering

Usage strategy:
- Call when the user explicitly says "remember…", "save this…", "keep in mind for next time…"
- Call when you have learned something that will clearly matter in future sessions
  (a stable preference, a constraint, a recurring context, biographical info)
- Do NOT use for ephemeral turn-level state — that is what the conversation history is for
- Do NOT use to remember tool outputs, file paths, or anything tied to this run
- The fact is scoped to the current user automatically; never pass or invent a user id

Returns: {"fact_id": "<uuid>"} — store the id if you may want to forget it later.

Examples:
- {"content": "Prefers dogs over cats"}
- {"content": "Allergic to peanuts", "metadata": {"category": "health"}}
- {"content": "Works in EST timezone", "metadata": {"category": "context"}}
"""

RECALL_DESCRIPTION = """Search the user's long-term memory for facts relevant to a query.

Key capabilities:
- Semantic search (not keyword) — matches meaning, paraphrases, synonyms
- Returns ranked results with similarity scores (1.0 = perfect match)
- Scoped to the current user automatically — never crosses users

Usage strategy:
- Call PROACTIVELY at the start of a turn when the request hints at something personal
  (preferences, past decisions, biographical info, recurring context)
- Call BEFORE answering questions where prior context would change your response
- A low score (< ~0.3) means weakly related — use judgment, do not blindly include
- Each result has a `fact_id` — keep it if you may want to forget that fact later
- Skip when the question is purely factual or has no user-specific component

Returns: list of {"fact_id": "<uuid>", "content": "<text>", "score": <float>}, most relevant first.

Examples:
- {"query": "food preferences"}
- {"query": "what does the user do for work?", "limit": 3}
- {"query": "timezone or schedule constraints", "limit": 10}
"""

FORGET_DESCRIPTION = """Delete a single fact from the user's long-term memory by id.

Key capabilities:
- Hard delete — the fact is gone, not just hidden
- Cross-user safety: returns "forbidden" if the fact belongs to another user
- Returns "not_found" if the id does not exist (already gone, or never existed)

Usage strategy:
- Call ONLY when the user explicitly asks to forget something, or you've discovered
  a fact is wrong and have no replacement to write
- ALWAYS get the fact_id from a prior `recall_facts` (or the original `remember_fact`
  response) — do NOT guess or fabricate an id
- To CORRECT a wrong fact, prefer `remember_fact` with the corrected content first
  (the old one stays, but new searches surface the correction); only call `forget_fact`
  if the user wants the stale fact removed

Returns: {"status": "deleted" | "not_found" | "forbidden"}.

Examples:
- {"fact_id": "8f1c2b40-9d3e-4a8c-9c1f-0d2e3a4b5c6d"}
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


class ForgetFactInputSchema(BaseModel):
    """LLM-visible input for `forget_fact`. `user_id` is bound at construction."""

    fact_id: str = Field(..., description="The id returned by recall_facts or remember_fact.")


class _LongTermMemoryTool(Node):
    """Shared base for the long-term memory tools."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    long_term_memory: LongTermMemory
    user_id: str


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

        fact = self.long_term_memory.remember(
            content=input_data.content,
            user_id=self.user_id,
            metadata=input_data.metadata,
        )
        return {"content": {"fact_id": fact.id}}


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
        return {
            "content": [{"fact_id": fact.id, "content": fact.content, "score": round(score, 4)} for fact, score in hits]
        }


class ForgetFactTool(_LongTermMemoryTool):
    """Delete a fact by id, with cross-user guard."""

    name: str = "forget_fact"
    description: str = FORGET_DESCRIPTION
    input_schema: ClassVar[type[ForgetFactInputSchema]] = ForgetFactInputSchema

    def execute(
        self, input_data: ForgetFactInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        logger.debug(f"Tool {self.name} - {self.id}: started")
        config = ensure_config(config)
        check_cancellation(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        status = self.long_term_memory.forget(
            fact_id=input_data.fact_id,
            user_id=self.user_id,
        )
        return {"content": {"status": status}}


_TOOL_BUILDERS: dict[MemoryToolKind, type[_LongTermMemoryTool]] = {
    MemoryToolKind.REMEMBER: RememberFactTool,
    MemoryToolKind.RECALL: RecallFactsTool,
    MemoryToolKind.FORGET: ForgetFactTool,
}


def build_long_term_memory_tools(
    *,
    long_term_memory: LongTermMemory,
    user_id: str,
    include: tuple[MemoryToolKind | str, ...] = (
        MemoryToolKind.REMEMBER,
        MemoryToolKind.RECALL,
        MemoryToolKind.FORGET,
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
