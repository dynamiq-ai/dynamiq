from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.runnables import RunnableConfig
from dynamiq.skills.sources.base import SkillSource
from dynamiq.utils.logger import logger


class SkillsToolInputSchema(BaseModel):
    """Input schema for Skills tool. Actions: list (discover), get (full or partial content)."""

    action: Literal["list", "get"] = Field(
        ...,
        description="Action: 'list' discover skills, 'get' full or partial skill content.",
    )
    skill_name: str | None = Field(default=None, description="Skill name (required for get)")
    section: str | None = Field(
        default=None,
        description="For get: return only this markdown section (e.g. 'Welcome messages')",
    )
    line_start: int | None = Field(default=None, description="For get: 1-based start line (body only)")
    line_end: int | None = Field(default=None, description="For get: 1-based end line (inclusive)")


class SkillsTool(Node):
    """Tool for skills: discover and get content from Dynamiq registry.

    Uses a configurable skill_source (Dynamiq API). After get, apply the
    skill's instructions yourself and provide the result in your final answer.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "SkillsTool"
    description: str = (
        "Manages skills (API-backed; instructions only). Use this tool to:\n"
        "- List available skills: action='list'\n"
        "- Get skill content: action='get', skill_name='...'. "
        "For large skills use section='Section title' or line_start/line_end to read only a part.\n\n"
        "After get, apply the skill's instructions yourself in your reasoning and provide the result "
        "in your final answer. Do not call the tool again with user content to transform; "
        "the tool only provides instructions; you produce the output."
    )

    skill_source: SkillSource = Field(..., description="Where skills are discovered and loaded from")
    input_schema: ClassVar[type[SkillsToolInputSchema]] = SkillsToolInputSchema

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {
            "skill_source": True,
        }

    def execute(
        self, input_data: SkillsToolInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        action = input_data.action
        logger.info("SkillsTool - action=%s", action)

        if action == "list":
            return self._list_skills()
        if action == "get":
            if not input_data.skill_name:
                raise ToolExecutionException("skill_name required for get", recoverable=True)
            return self._get_skill(
                input_data.skill_name,
                section=input_data.section,
                line_start=input_data.line_start,
                line_end=input_data.line_end,
            )
        raise ToolExecutionException(f"Unknown action: {action}", recoverable=True)

    def _list_skills(self) -> dict[str, Any]:
        skills = self.skill_source.discover_skills()
        skills_info = []
        for s in skills:
            info = {"name": s.name, "description": s.description, "tags": s.tags}
            if s.id is not None:
                info["id"] = s.id
            if s.version_id is not None:
                info["version_id"] = s.version_id
            skills_info.append(info)
        names = [s.name for s in skills]
        logger.info("SkillsTool - list: %d skill(s) %s", len(skills), names)
        return {
            "content": {
                "available_skills": skills_info,
                "total": len(skills),
            }
        }

    def _get_skill(
        self,
        skill_name: str,
        section: str | None = None,
        line_start: int | None = None,
        line_end: int | None = None,
    ) -> dict[str, Any]:
        if section is not None or line_start is not None or line_end is not None:
            out = self.skill_source.load_skill_content(
                skill_name,
                section=section,
                line_start=line_start,
                line_end=line_end,
            )
            if not out:
                raise ToolExecutionException(f"Skill '{skill_name}' not found", recoverable=True)
            logger.info(
                "SkillsTool - get: skill=%s (section=%s, lines=%s-%s) -> content received",
                skill_name,
                section,
                line_start,
                line_end,
            )
            return {"content": out}
        skill = self.skill_source.load_skill(skill_name)
        if not skill:
            raise ToolExecutionException(f"Skill '{skill_name}' not found", recoverable=True)
        content = skill.get_full_content()
        logger.info("SkillsTool - get: skill=%s -> content received (%d chars)", skill_name, len(content))
        return {
            "content": {
                "skill_name": skill.name,
                "description": skill.metadata.description,
                "instructions": content,
                "supporting_files": [str(p) for p in skill.supporting_files_paths],
                "dependencies": skill.metadata.dependencies,
            }
        }
