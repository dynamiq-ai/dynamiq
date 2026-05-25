"""Tools giving an agent access to a `LongTermMemory` instance.

The three tools (`remember_fact`, `recall_facts`, `forget_fact`) bind
`user_id` at construction. `user_id` never appears in `InputSchema`, so
the model has no slot to address another user's memory. See spec §9.1.
"""
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.memory.long_term import LongTermMemory
from dynamiq.nodes.node import Node
from dynamiq.nodes.types import NodeGroup


REMEMBER_DESCRIPTION = (
    "Record a durable fact about the current user that should persist across "
    "conversations (preferences, constraints, recurring context, biographical info). "
    "Use only when you've learned something that will matter in future sessions — "
    "not for ephemeral turn-level state. Returns {fact_id: <uuid>}. Calling twice "
    "with the same content returns the same fact_id."
)

RECALL_DESCRIPTION = (
    "Search the user's long-term memory for facts relevant to a query. "
    "Use BEFORE answering questions where prior context (preferences, past "
    "decisions, biographical info) would change the response. Returns a list of "
    "{fact_id, content, score} entries, most relevant first."
)

FORGET_DESCRIPTION = (
    "Delete a fact from the user's long-term memory by id. Use when the user "
    "explicitly asks to be forgotten on something, or when a fact is wrong and "
    "you have no replacement. Get the fact_id from a prior recall_facts call. "
    "Returns {status: 'deleted'|'not_found'|'forbidden'}."
)


class RememberFactInputSchema(BaseModel):
    """LLM-visible input for remember_fact. Note: no `user_id`."""

    content: str = Field(..., min_length=1, max_length=1000,
                         description="The fact to remember, as a short statement.")
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional free-form metadata (e.g. {'category': 'preference'}).",
    )


class RecallFactsInputSchema(BaseModel):
    """LLM-visible input for recall_facts. Note: no `user_id`."""

    query: str = Field(..., min_length=1, max_length=500,
                       description="What to search for.")
    limit: int = Field(default=5, ge=1, le=20,
                       description="Max facts to return.")


class ForgetFactInputSchema(BaseModel):
    """LLM-visible input for forget_fact. Note: no `user_id`."""

    fact_id: str = Field(..., description="The id returned by recall_facts or remember_fact.")


class _LongTermMemoryTool(Node):
    """Shared base for the three long-term memory tools.

    Holds the `LongTermMemory` reference and the construction-bound `user_id`.
    Concrete subclasses set `name`, `description`, `input_schema`, and `execute`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    long_term_memory: LongTermMemory
    user_id: str


class RememberFactTool(_LongTermMemoryTool):
    """Write a fact to long-term memory, scoped to the bound user_id."""

    name: str = "remember_fact"
    description: str = REMEMBER_DESCRIPTION
    input_schema: ClassVar[type[RememberFactInputSchema]] = RememberFactInputSchema

    def execute(self, input_data: RememberFactInputSchema, config=None, **kwargs) -> dict:
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

    def execute(self, input_data: RecallFactsInputSchema, config=None, **kwargs) -> dict:
        hits = self.long_term_memory.recall(
            query=input_data.query,
            user_id=self.user_id,
            limit=input_data.limit,
        )
        return {
            "content": [
                {"fact_id": fact.id, "content": fact.content,
                 "score": round(score, 4)}
                for fact, score in hits
            ]
        }


class ForgetFactTool(_LongTermMemoryTool):
    """Delete a fact by id, with cross-user guard."""

    name: str = "forget_fact"
    description: str = FORGET_DESCRIPTION
    input_schema: ClassVar[type[ForgetFactInputSchema]] = ForgetFactInputSchema

    def execute(self, input_data: ForgetFactInputSchema, config=None, **kwargs) -> dict:
        status = self.long_term_memory.forget(
            fact_id=input_data.fact_id,
            user_id=self.user_id,
        )
        return {"content": {"status": status}}


_TOOL_BUILDERS: dict[str, type[_LongTermMemoryTool]] = {
    "remember": RememberFactTool,
    "recall": RecallFactsTool,
    "forget": ForgetFactTool,
}


def build_long_term_memory_tools(
    *,
    long_term_memory: LongTermMemory,
    user_id: str,
    include: tuple[str, ...] = ("remember", "recall", "forget"),
) -> list[Node]:
    """Build the long-term-memory tools with `user_id` baked in.

    `include` selects which tools to return — sub-agents commonly use
    `include=("recall",)` for read-only inheritance. Unknown keys are ignored.
    """
    tools: list[Node] = []
    for kind in include:
        cls = _TOOL_BUILDERS.get(kind)
        if cls is None:
            continue
        tools.append(cls(long_term_memory=long_term_memory, user_id=user_id))
    return tools
