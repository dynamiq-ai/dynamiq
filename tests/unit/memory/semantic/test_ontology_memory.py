import json
from datetime import datetime, timezone

from dynamiq import connections
from dynamiq.memory.semantic import ContextRetrievalMode, OntologyMemory
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.tools.ontology_memory import OntologyMemoryTool, OntologyMemoryToolInputSchema
from dynamiq.ontology import EpisodeSourceType, OntologyEntityType, Provenance, TemporalFact
from dynamiq.runnables import RunnableResult, RunnableStatus
from dynamiq.storages.graph import InMemoryGraphStore


def make_mock_extraction_llm(mocker, mapping: dict[str, dict]):
    llm = OpenAI(model="gpt-4o-mini", connection=connections.OpenAI(api_key="test"))

    def _run(*args, **kwargs):
        prompt = kwargs["prompt"]
        content = prompt.messages[-1].content
        for marker, payload in mapping.items():
            if marker in content:
                return RunnableResult(
                    status=RunnableStatus.SUCCESS,
                    input={},
                    output={"content": json.dumps(payload)},
                )
        raise AssertionError(f"No mock extraction payload for prompt: {content}")

    mock = mocker.patch.object(llm, "run", side_effect=_run)
    return llm, mock


def test_ontology_memory_persists_and_queries_entity_fact_graph():
    store = InMemoryGraphStore()
    memory = OntologyMemory(graph_store=store)

    episode = memory.add_episode(
        content="Alex prefers concise technical answers.",
        source_type=EpisodeSourceType.MESSAGE,
        source_id="msg-1",
        user_id="user-1",
        session_id="session-1",
    )
    user = memory.resolve_or_create_entity(label="Alex", entity_type=OntologyEntityType.USER, aliases=["Alexander"])
    preference = memory.resolve_or_create_entity(
        label="Concise technical answers",
        entity_type=OntologyEntityType.PREFERENCE,
    )
    valid_at = datetime(2026, 5, 14, 9, 0, tzinfo=timezone.utc)
    fact = memory.add_fact(
        TemporalFact(
            id="fact-1",
            subject_id=user.id,
            predicate="has_preference",
            object_id=preference.id,
            valid_at=valid_at,
            episode_ids=[episode.id],
            user_id="user-1",
            session_id="session-1",
        )
    )

    results = memory.get_facts(
        subject_id=user.id,
        predicate="has_preference",
        valid_at=datetime(2026, 5, 14, 10, 0, tzinfo=timezone.utc),
    )

    assert fact.id in store.facts
    assert store.fact_episodes[fact.id] == [episode.id]
    assert len(results) == 1
    assert results[0].id == "fact-1"
    assert results[0].object_id == preference.id


def test_ontology_memory_supports_literal_fact_and_provenance():
    store = InMemoryGraphStore()
    memory = OntologyMemory(graph_store=store)

    episode = memory.add_episode(
        content="Alex prefers concise answers.",
        source_type=EpisodeSourceType.MESSAGE,
        source_id="msg-2",
        user_id="user-1",
    )
    user = memory.resolve_or_create_entity(label="Alex", entity_type=OntologyEntityType.USER)
    fact = memory.add_fact(
        TemporalFact(
            id="fact-2",
            subject_id=user.id,
            predicate="has_preference",
            object_value="concise answers",
            episode_ids=[episode.id],
            user_id="user-1",
        )
    )
    provenance = memory.add_provenance(
        Provenance(id="prov-1", episode_id=episode.id, fact_id=fact.id, extraction_model="gpt-test")
    )

    results = memory.get_facts(subject_id=user.id, predicate="has_preference")
    audit = memory.audit_fact(fact_id=fact.id)

    assert provenance.id in store.provenance
    assert results[0].object_value == "concise answers"
    assert audit["provenance"][0].id == provenance.id


def test_ontology_memory_persists_local_graph_state(tmp_path):
    state_file = tmp_path / "ontology-state.json"
    first_memory = OntologyMemory(graph_store=InMemoryGraphStore(state_file=str(state_file)))

    episode = first_memory.add_episode(
        content="Alex likes pizza.",
        source_type=EpisodeSourceType.MESSAGE,
        user_id="user-1",
        session_id="session-1",
    )
    user = first_memory.resolve_or_create_entity(label="Alex", entity_type=OntologyEntityType.USER)
    pizza = first_memory.resolve_or_create_entity(label="pizza", entity_type=OntologyEntityType.CONCEPT)
    first_memory.link_episode_entity(episode_id=episode.id, entity_id=user.id)
    first_memory.link_episode_entity(episode_id=episode.id, entity_id=pizza.id)
    first_memory.add_fact(
        TemporalFact(
            subject_id=user.id,
            predicate="likes",
            object_id=pizza.id,
            episode_ids=[episode.id],
            user_id="user-1",
            session_id="session-1",
        )
    )

    second_memory = OntologyMemory(graph_store=InMemoryGraphStore(state_file=str(state_file)))
    facts = second_memory.search_facts(user_id="user-1", session_id="session-1")
    entities = second_memory.get_episode_entities(episode_id=episode.id)

    assert len(facts) == 1
    assert facts[0]["predicate"] == "likes"
    assert {entity.label for entity in entities} == {"Alex", "pizza"}


def test_extract_and_commit_invalidates_older_conflicting_fact(mocker):
    llm, _ = make_mock_extraction_llm(
        mocker,
        {
            "I prefer concise answers.": {
                "entities": [
                    {"label": "Alex", "entity_type": "User", "aliases": [], "confidence": 0.98},
                    {"label": "concise answers", "entity_type": "Preference", "aliases": [], "confidence": 0.94},
                ],
                "facts": [
                    {
                        "subject_label": "Alex",
                        "predicate": "has_preference",
                        "object_label": "concise answers",
                        "subject_type": "User",
                        "object_type": "Preference",
                        "confidence": 0.94,
                    }
                ],
                "notes": [],
            },
            "I now prefer detailed answers.": {
                "entities": [
                    {"label": "Alex", "entity_type": "User", "aliases": [], "confidence": 0.98},
                    {"label": "detailed answers", "entity_type": "Preference", "aliases": [], "confidence": 0.94},
                ],
                "facts": [
                    {
                        "subject_label": "Alex",
                        "predicate": "has_preference",
                        "object_label": "detailed answers",
                        "subject_type": "User",
                        "object_type": "Preference",
                        "confidence": 0.94,
                    }
                ],
                "notes": [],
            },
        },
    )
    memory = OntologyMemory(graph_store=InMemoryGraphStore(), llm=llm)

    first_episode = memory.add_episode(
        content="I prefer concise answers.",
        source_type=EpisodeSourceType.MESSAGE,
        user_id="user-1",
        session_id="session-1",
        metadata={"user_label": "Alex"},
        observed_at=datetime(2026, 5, 14, 9, 0, tzinfo=timezone.utc),
    )
    second_episode = memory.add_episode(
        content="I now prefer detailed answers.",
        source_type=EpisodeSourceType.MESSAGE,
        user_id="user-1",
        session_id="session-1",
        metadata={"user_label": "Alex"},
        observed_at=datetime(2026, 5, 14, 10, 0, tzinfo=timezone.utc),
    )

    memory.extract_and_commit(episode=first_episode)
    memory.extract_and_commit(episode=second_episode)

    active = memory.get_facts(user_id="user-1", session_id="session-1", include_inactive=False, limit=10)
    all_facts = memory.get_facts(user_id="user-1", session_id="session-1", include_inactive=True, limit=10)

    assert len(active) == 1
    assert active[0].object_id is not None
    assert len(all_facts) == 2
    invalidated = [fact for fact in all_facts if fact.status != "active"]
    assert len(invalidated) == 1
    assert invalidated[0].invalid_at == second_episode.observed_at


def test_context_block_includes_relevant_facts_and_episodes(mocker):
    llm, _ = make_mock_extraction_llm(
        mocker,
        {
            "I work at OpenAI.": {
                "entities": [
                    {"label": "Alex", "entity_type": "User", "aliases": [], "confidence": 0.98},
                    {"label": "OpenAI", "entity_type": "Organization", "aliases": [], "confidence": 0.96},
                ],
                "facts": [
                    {
                        "subject_label": "Alex",
                        "predicate": "works_at",
                        "object_label": "OpenAI",
                        "subject_type": "User",
                        "object_type": "Organization",
                        "confidence": 0.96,
                    }
                ],
                "notes": [],
            }
        },
    )
    memory = OntologyMemory(graph_store=InMemoryGraphStore(), llm=llm)
    episode = memory.add_episode(
        content="I work at OpenAI.",
        source_type=EpisodeSourceType.MESSAGE,
        user_id="user-1",
        session_id="session-1",
        metadata={"user_label": "Alex"},
    )
    memory.extract_and_commit(episode=episode)

    block = memory.get_context_block(
        query="Where does Alex work?",
        user_id="user-1",
        session_id="session-1",
        mode=ContextRetrievalMode.CURRENT,
    )

    assert "works_at" in block
    assert "OpenAI" in block
    assert "Recent Episodes" in block


def test_historical_context_block_filters_stale_facts(mocker):
    llm, _ = make_mock_extraction_llm(
        mocker,
        {
            "I prefer concise answers.": {
                "entities": [
                    {"label": "Alex", "entity_type": "User", "aliases": [], "confidence": 0.98},
                    {"label": "concise answers", "entity_type": "Preference", "aliases": [], "confidence": 0.94},
                ],
                "facts": [
                    {
                        "subject_label": "Alex",
                        "predicate": "has_preference",
                        "object_label": "concise answers",
                        "subject_type": "User",
                        "object_type": "Preference",
                        "confidence": 0.94,
                    }
                ],
                "notes": [],
            },
            "I now prefer detailed answers.": {
                "entities": [
                    {"label": "Alex", "entity_type": "User", "aliases": [], "confidence": 0.98},
                    {"label": "detailed answers", "entity_type": "Preference", "aliases": [], "confidence": 0.94},
                ],
                "facts": [
                    {
                        "subject_label": "Alex",
                        "predicate": "has_preference",
                        "object_label": "detailed answers",
                        "subject_type": "User",
                        "object_type": "Preference",
                        "confidence": 0.94,
                    }
                ],
                "notes": [],
            },
        },
    )
    memory = OntologyMemory(graph_store=InMemoryGraphStore(), llm=llm)

    first_time = datetime(2026, 5, 14, 9, 0, tzinfo=timezone.utc)
    second_time = datetime(2026, 5, 14, 10, 0, tzinfo=timezone.utc)
    first_episode = memory.add_episode(
        content="I prefer concise answers.",
        source_type=EpisodeSourceType.MESSAGE,
        user_id="user-1",
        session_id="session-1",
        metadata={"user_label": "Alex"},
        observed_at=first_time,
    )
    second_episode = memory.add_episode(
        content="I now prefer detailed answers.",
        source_type=EpisodeSourceType.MESSAGE,
        user_id="user-1",
        session_id="session-1",
        metadata={"user_label": "Alex"},
        observed_at=second_time,
    )
    memory.extract_and_commit(episode=first_episode)
    memory.extract_and_commit(episode=second_episode)

    historical = memory.get_context_block(
        query="What did Alex prefer earlier?",
        user_id="user-1",
        session_id="session-1",
        mode=ContextRetrievalMode.HISTORICAL,
        valid_at=datetime(2026, 5, 14, 9, 30, tzinfo=timezone.utc),
    )

    assert "concise answers" in historical
    assert "detailed answers" not in historical


def test_context_block_includes_entities_extracted_without_facts(mocker):
    llm, _ = make_mock_extraction_llm(
        mocker,
        {
            "im alex": {
                "entities": [
                    {"label": "Alex", "entity_type": "User", "aliases": [], "confidence": 0.98},
                ],
                "facts": [],
                "notes": [],
            }
        },
    )
    memory = OntologyMemory(graph_store=InMemoryGraphStore(), llm=llm)
    episode = memory.add_episode(
        content="im alex",
        source_type=EpisodeSourceType.MESSAGE,
        user_id="user-1",
        session_id="session-1",
        metadata={"user_label": "Alex"},
    )
    memory.extract_and_commit(episode=episode)

    block = memory.get_context_block(
        query="What should the assistant remember about this user?",
        user_id="user-1",
        session_id="session-1",
    )

    assert "No matching facts found." in block
    assert "User: Alex" in block


def test_ontology_memory_tool_add_episode_and_context_block(mocker):
    llm, run_mock = make_mock_extraction_llm(
        mocker,
        {
            "I prefer concise technical answers.": {
                "entities": [
                    {"label": "Alex", "entity_type": "User", "aliases": [], "confidence": 0.98},
                    {
                        "label": "concise technical answers",
                        "entity_type": "Preference",
                        "aliases": [],
                        "confidence": 0.95,
                    },
                ],
                "facts": [
                    {
                        "subject_label": "Alex",
                        "predicate": "has_preference",
                        "object_label": "concise technical answers",
                        "subject_type": "User",
                        "object_type": "Preference",
                        "confidence": 0.95,
                    }
                ],
                "notes": [],
            }
        },
    )
    tool = OntologyMemoryTool(client=object(), graph_store=InMemoryGraphStore(), llm=llm)
    tool.init_components()

    add_result = tool.execute(
        OntologyMemoryToolInputSchema(
            operation="add_episode",
            content="I prefer concise technical answers.",
            user_id="user-1",
            session_id="session-1",
            metadata={"user_label": "Alex"},
        )
    )
    context_result = tool.execute(
        OntologyMemoryToolInputSchema(
            operation="get_context_block",
            query="How should the assistant answer this user?",
            user_id="user-1",
            session_id="session-1",
        )
    )

    assert add_result["commit"]["facts"]
    assert "concise technical answers" in context_result["context_block"]
    response_format = run_mock.call_args.kwargs["response_format"]
    schema = response_format["json_schema"]["schema"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "ExtractionResult"
    assert response_format["json_schema"]["strict"] is True
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["entities", "facts", "notes"]
    assert schema["$defs"]["ExtractedEntity"]["additionalProperties"] is False
    assert schema["$defs"]["ExtractedEntity"]["required"] == [
        "label",
        "entity_type",
        "aliases",
        "summary",
        "confidence",
    ]
    assert schema["$defs"]["ExtractedFact"]["additionalProperties"] is False
    assert schema["$defs"]["ExtractedFact"]["required"] == [
        "subject_label",
        "predicate",
        "object_label",
        "object_value",
        "subject_type",
        "object_type",
        "confidence",
    ]
