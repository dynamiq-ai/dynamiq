from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from dynamiq.connections import ApacheAGE, AWSNeptune, Neo4j
from dynamiq.connections.managers import ConnectionManager
from dynamiq.memory.semantic.memory import OntologyMemory
from dynamiq.memory.semantic.retrieval import ContextRetrievalMode
from dynamiq.nodes import ErrorHandling, NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.llms.base import BaseLLM
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.nodes.types import ActionType
from dynamiq.ontology import EpisodeSourceType
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.graph.age import ApacheAgeGraphStore
from dynamiq.storages.graph.base import BaseGraphStore
from dynamiq.storages.graph.neo4j import Neo4jGraphStore
from dynamiq.storages.graph.neptune import NeptuneGraphStore
from dynamiq.utils.logger import logger


class OntologyMemoryToolInputSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: Literal[
        "add_episode",
        "search_facts",
        "get_entity_context",
        "get_context_block",
        "audit_fact",
    ] = Field(description="Ontology memory operation to execute.")
    content: str | None = Field(default=None, description="Episode content for add_episode.")
    source_type: EpisodeSourceType = Field(default=EpisodeSourceType.MESSAGE, description="Episode source type.")
    source_id: str | None = Field(default=None, description="Optional external source id.")
    actor_id: str | None = Field(default=None, description="Actor id associated with the episode.")
    user_id: str | None = Field(default=None, description="User scope for facts and episodes.")
    session_id: str | None = Field(default=None, description="Session scope for facts and episodes.")
    workflow_id: str | None = Field(default=None, description="Workflow scope for facts and episodes.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Extra episode metadata.")
    auto_extract: bool = Field(default=True, description="Run extraction and commit immediately after add_episode.")
    query: str | None = Field(default=None, description="Search or context query.")
    predicate: str | None = Field(default=None, description="Optional fact predicate filter.")
    entity_id: str | None = Field(default=None, description="Entity id for get_entity_context.")
    entity_label: str | None = Field(default=None, description="Entity label for get_entity_context.")
    fact_id: str | None = Field(default=None, description="Fact id for audit_fact.")
    include_inactive: bool = Field(default=False, description="Include invalidated or rejected facts.")
    mode: ContextRetrievalMode = Field(default=ContextRetrievalMode.CURRENT, description="Context retrieval mode.")
    valid_at: datetime | None = Field(default=None, description="Historical point-in-time query.")
    limit: int | None = Field(default=None, ge=1, description="Result limit override.")


class OntologyMemoryTool(ConnectionNode):
    """Agent-facing ontology memory tool with graph-backed persistence."""

    input_schema: ClassVar[type[OntologyMemoryToolInputSchema]] = OntologyMemoryToolInputSchema

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    action_type: ActionType = ActionType.DATABASE_QUERY
    name: str = "ontology-memory"
    description: str = """Ontology-aware memory tool for durable context.

Supported operations:
- add_episode: persist a source episode and optionally extract entities/facts immediately
- search_facts: search accepted facts scoped by user/session/query
- get_entity_context: return one entity with its related facts and source episodes
- get_context_block: build a compact memory block for an agent prompt
- audit_fact: inspect one fact with provenance and source episodes

Usage notes:
- Prefer add_episode with auto_extract=true for new user messages or tool outputs
- Pass user_id and session_id consistently to keep memory scoped
- Use get_context_block before answering when durable context may matter
"""
    error_handling: ErrorHandling = Field(default_factory=lambda: ErrorHandling(timeout_seconds=600))
    connection: Neo4j | ApacheAGE | AWSNeptune | None = None
    graph_name: str | None = None
    create_graph_if_not_exists: bool = False
    ensure_schema_on_init: bool = True
    memory_query_limit: int = Field(default=100, gt=0)
    graph_store: BaseGraphStore | None = None
    llm: BaseLLM | None = None

    _graph_store: BaseGraphStore | None = PrivateAttr(default=None)
    _memory: OntologyMemory | None = PrivateAttr(default=None)

    @property
    def to_dict_exclude_params(self) -> dict:
        return super().to_dict_exclude_params | {"graph_store": True, "llm": True}

    def to_dict(self, **kwargs) -> dict:
        data = super().to_dict(**kwargs)
        data["llm"] = self.llm.to_dict(**kwargs) if self.llm is not None and hasattr(self.llm, "to_dict") else None
        return data

    def init_components(self, connection_manager: ConnectionManager | None = None) -> None:
        super().init_components(connection_manager)
        if self.graph_store is not None:
            self._graph_store = self.graph_store
        elif isinstance(self.connection, ApacheAGE):
            self._graph_store = ApacheAgeGraphStore(
                connection=self.connection,
                client=self.client,
                graph_name=self.graph_name,
                create_graph_if_not_exists=self.create_graph_if_not_exists,
            )
        elif isinstance(self.connection, AWSNeptune):
            self._graph_store = NeptuneGraphStore(
                connection=self.connection,
                client=self.client,
                endpoint=self.connection.endpoint,
                verify_ssl=self.connection.verify_ssl,
                timeout=self.connection.timeout,
            )
        elif isinstance(self.connection, Neo4j):
            self._graph_store = Neo4jGraphStore(connection=self.connection, client=self.client)

        if self._graph_store is None:
            raise ValueError("OntologyMemoryTool requires either graph_store or a supported graph connection.")

        if self.llm is not None and self.llm.is_postponed_component_init:
            self.llm.init_components(connection_manager)

        self._memory = OntologyMemory(
            graph_store=self._graph_store,
            query_limit=self.memory_query_limit,
            llm=self.llm,
        )
        if self.ensure_schema_on_init:
            self._memory.ensure_schema()

    def ensure_client(self) -> None:
        previous_client = self.client
        super().ensure_client()
        if self.client is previous_client or self._graph_store is None:
            return
        graph_client = getattr(self._graph_store, "client", None)
        if graph_client is not self.client:
            self._graph_store.update_client(self.client)

    def execute(
        self, input_data: OntologyMemoryToolInputSchema, config: RunnableConfig = None, **kwargs
    ) -> dict[str, Any]:
        logger.info("Tool %s - %s: started with INPUT DATA:\n%s", self.name, self.id, input_data.model_dump())
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        if self._memory is None:
            raise ToolExecutionException("Ontology memory is not initialized.", recoverable=True)

        try:
            result = self._dispatch(input_data)
        except Exception as exc:  # noqa: BLE001
            logger.error("Tool %s - %s: failed: %s", self.name, self.id, exc)
            raise ToolExecutionException(str(exc), recoverable=True) from exc

        logger.info("Tool %s - %s: finished successfully.", self.name, self.id)
        return result

    def _dispatch(self, input_data: OntologyMemoryToolInputSchema) -> dict[str, Any]:
        if input_data.operation == "add_episode":
            if not input_data.content:
                raise ValueError("content is required for add_episode.")
            episode_kwargs = {
                "content": input_data.content,
                "source_type": input_data.source_type,
                "actor_id": input_data.actor_id,
                "user_id": input_data.user_id,
                "session_id": input_data.session_id,
                "workflow_id": input_data.workflow_id,
                "metadata": input_data.metadata,
            }
            if input_data.source_id is not None:
                episode_kwargs["source_id"] = input_data.source_id
            episode = self._memory.add_episode(**episode_kwargs)
            payload: dict[str, Any] = {"episode": self._serialize(episode)}
            if input_data.auto_extract:
                committed = self._memory.extract_and_commit(episode=episode)
                payload["commit"] = self._serialize(committed)
                payload["content"] = (
                    f"Stored episode {episode.id} and committed {len(committed['facts'])} facts "
                    f"from {len(committed['entities'])} entities."
                )
            else:
                payload["content"] = f"Stored episode {episode.id}."
            return payload

        if input_data.operation == "search_facts":
            facts = self._memory.search_facts(
                query=input_data.query,
                user_id=input_data.user_id,
                session_id=input_data.session_id,
                predicate=input_data.predicate,
                include_inactive=input_data.include_inactive,
                limit=input_data.limit,
            )
            return {
                "facts": self._serialize(facts),
                "content": f"Found {len(facts)} facts.",
            }

        if input_data.operation == "get_entity_context":
            if not input_data.entity_id and not input_data.entity_label:
                raise ValueError("entity_id or entity_label is required for get_entity_context.")
            context = self._memory.get_entity_context(
                entity_id=input_data.entity_id, entity_label=input_data.entity_label
            )
            entity = context["entity"]
            return {
                "entity_context": self._serialize(context),
                "content": f"Loaded context for entity {entity.label} ({entity.id}).",
            }

        if input_data.operation == "get_context_block":
            block = self._memory.get_context_block(
                query=input_data.query,
                user_id=input_data.user_id,
                session_id=input_data.session_id,
                mode=input_data.mode,
                valid_at=input_data.valid_at,
                limit=input_data.limit,
            )
            return {"context_block": block, "content": block}

        if input_data.operation == "audit_fact":
            if not input_data.fact_id:
                raise ValueError("fact_id is required for audit_fact.")
            audit = self._memory.audit_fact(fact_id=input_data.fact_id)
            return {
                "audit": self._serialize(audit),
                "content": f"Audited fact {input_data.fact_id}.",
            }

        raise ValueError(f"Unsupported operation: {input_data.operation}")

    @staticmethod
    def _serialize(value: Any) -> Any:
        if isinstance(value, BaseModel):
            return {key: OntologyMemoryTool._serialize(item) for key, item in value.model_dump().items()}
        if isinstance(value, dict):
            return {key: OntologyMemoryTool._serialize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [OntologyMemoryTool._serialize(item) for item in value]
        return value
