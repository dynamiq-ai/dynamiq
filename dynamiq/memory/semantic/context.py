from __future__ import annotations

from typing import Any

from dynamiq.ontology import Entity, Episode


class ContextBlockBuilder:
    """Build deterministic prompt-ready context blocks from ontology memory."""

    def build(
        self,
        *,
        query: str | None,
        entities: list[Entity],
        facts: list[dict[str, Any]],
        episodes: list[Episode],
    ) -> str:
        lines: list[str] = []
        if query:
            lines.append("## Query")
            lines.append(query)
            lines.append("")

        lines.append("## Active Facts")
        if facts:
            for fact in facts:
                subject = fact.get("subject_label") or fact.get("subject_id")
                obj = fact.get("object_label") or fact.get("object_value") or fact.get("object_id")
                lines.append(
                    f"- {subject} --{fact.get('predicate')}--> {obj} "
                    f"[confidence={fact.get('confidence')}, episode_ids={fact.get('episode_ids', [])}]"
                )
        else:
            lines.append("- No matching facts found.")
        lines.append("")

        lines.append("## Relevant Entities")
        if entities:
            for entity in entities:
                aliases = f" aliases={entity.aliases}" if entity.aliases else ""
                lines.append(f"- {entity.entity_type.value}: {entity.label}{aliases}")
        else:
            lines.append("- No relevant entities found.")
        lines.append("")

        lines.append("## Recent Episodes")
        if episodes:
            for episode in episodes[:5]:
                lines.append(f"- [{episode.observed_at.isoformat()}] {episode.source_type.value}: {episode.content}")
        else:
            lines.append("- No recent episodes found.")

        return "\n".join(lines)
