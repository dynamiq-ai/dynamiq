import json

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.memory.semantic import OntologyMemory
from dynamiq.nodes.llms import OpenAI
from dynamiq.runnables import RunnableResult, RunnableStatus
from dynamiq.storages.graph import InMemoryGraphStore
from dynamiq.utils.logger import logger


def build_mock_llm() -> OpenAI:
    llm = OpenAI(model="gpt-4o-mini", connection=OpenAIConnection(api_key="test"))

    def _mock_run(*args, **kwargs):
        prompt = kwargs["prompt"]
        content = prompt.messages[-1].content
        if "prefer concise technical answers" in content:
            payload = {
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
                "notes": ["mock extraction"],
            }
        elif "work at OpenAI" in content:
            payload = {
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
                "notes": ["mock extraction"],
            }
        else:
            payload = {
                "entities": [
                    {"label": "Alex", "entity_type": "User", "aliases": [], "confidence": 0.98},
                    {
                        "label": "detailed technical answers when debugging",
                        "entity_type": "Preference",
                        "aliases": [],
                        "confidence": 0.95,
                    },
                ],
                "facts": [
                    {
                        "subject_label": "Alex",
                        "predicate": "has_preference",
                        "object_label": "detailed technical answers when debugging",
                        "subject_type": "User",
                        "object_type": "Preference",
                        "confidence": 0.95,
                    }
                ],
                "notes": ["mock extraction"],
            }

        return RunnableResult(
            status=RunnableStatus.SUCCESS,
            input={},
            output={"content": json.dumps(payload)},
        )

    llm.run = _mock_run
    return llm


def main() -> None:
    memory = OntologyMemory(graph_store=InMemoryGraphStore(), llm=build_mock_llm())

    episodes = [
        "I prefer concise technical answers.",
        "I work at OpenAI.",
        "I now prefer detailed technical answers when debugging.",
    ]

    for index, content in enumerate(episodes, start=1):
        logger.info("Adding episode %s: %s", index, content)
        episode = memory.add_episode(
            content=content,
            source_type="message",
            source_id=f"demo-{index}",
            user_id="demo-user",
            session_id="demo-session",
            metadata={"user_label": "Alex"},
        )
        commit = memory.extract_and_commit(episode=episode)
        logger.info(
            "Committed episode %s with %s entities and %s facts.",
            episode.id,
            len(commit["entities"]),
            len(commit["facts"]),
        )

    print("=== FACTS ===")
    for fact in memory.search_facts(user_id="demo-user", session_id="demo-session", include_inactive=True):
        print(fact)

    print("\n=== CONTEXT BLOCK ===")
    print(
        memory.get_context_block(
            query="How should the assistant answer this user while debugging?",
            user_id="demo-user",
            session_id="demo-session",
        )
    )


if __name__ == "__main__":
    main()
