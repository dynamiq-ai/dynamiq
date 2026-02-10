"""Unit tests for SkillsTool behavior."""

import pytest

from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.tools.skills_tool import SkillsTool, SkillsToolInputSchema
from dynamiq.skills.registries.base import BaseSkillRegistry
from dynamiq.skills.types import SkillInstructions, SkillMetadata


class _DummyRegistry(BaseSkillRegistry):
    def get_skills_metadata(self) -> list[SkillMetadata]:
        return [SkillMetadata(name="demo", description="Demo skill")]

    def get_skill_instructions(self, name: str) -> SkillInstructions:
        return SkillInstructions(
            name=name,
            description="Demo skill",
            instructions="# Intro\nhello\n## Details\nworld\n",
        )


def test_skills_tool_get_missing_section_raises_recoverable_error():
    tool = SkillsTool(skill_registry=_DummyRegistry())

    with pytest.raises(ToolExecutionException, match="Section 'Missing' not found"):
        tool.execute(
            SkillsToolInputSchema(
                action="get",
                skill_name="demo",
                section="Missing",
            )
        )
