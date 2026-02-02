import io
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.runnables import RunnableConfig
from dynamiq.skills.executor import SkillExecutor
from dynamiq.skills.sources.base import SkillSource
from dynamiq.utils.logger import logger


class SkillsToolInputSchema(BaseModel):
    """Input schema for Skills tool.

    Actions: list (discover), get (full or partial content), run_script (sandbox).
    For large skills use section or line_start/line_end to read only what you need.
    """

    action: Literal["list", "get", "run_script"] = Field(
        ...,
        description=(
            "Action: 'list' discover skills, 'get' full or partial skill content, "
            "'run_script' execute a skill script in the sandbox"
        )
    )
    skill_name: str | None = Field(default=None, description="Skill name (required for get and run_script)")
    section: str | None = Field(
        default=None, description="For get: return only this markdown section (e.g. 'Welcome messages')"
    )
    line_start: int | None = Field(default=None, description="For get: 1-based start line (body only)")
    line_end: int | None = Field(default=None, description="For get: 1-based end line (inclusive)")
    script_path: str | None = Field(
        default=None, description="Path relative to skill root, e.g. scripts/script.py (required for run_script)"
    )
    arguments: list[str] | None = Field(default=None, description="Arguments for the script (for run_script)")
    input_files: dict[str, str] | None = Field(
        default=None,
        description="For run_script: map FileStore path -> sandbox path (e.g. {'data/in.html': 'input/in.html'})",
    )
    output_paths: list[str] | None = Field(
        default=None, description="For run_script: sandbox paths to collect after run (e.g. ['output/out.pptx'])"
    )
    output_prefix: str = Field(
        default="", description="For run_script: FileStore path prefix for collected files (e.g. 'generated/')"
    )


class SkillsTool(Node):
    """Tool for skills: discover, get content, run scripts.

    Uses a configurable skill_source (e.g. FileStore, E2B). run_script uses
    skill_executor when set (e.g. subprocess sandbox, E2B).
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "SkillsTool"
    description: str = (
        "Manages skills. Use this tool to:\n"
        "- List available skills: action='list'\n"
        "- Get skill content: action='get', skill_name='...'. "
        "For large skills use section='Section title' or line_start/line_end to read only a part.\n"
        "- Run a skill script in the sandbox: action='run_script', "
        "skill_name='...', script_path='scripts/...', optional arguments=[]\n\n"
        "Do not load full skill content until you need it; "
        "use list first, then get (or get with section/lines) when the task requires it."
    )

    skill_source: SkillSource = Field(..., description="Where skills are discovered and loaded from")
    skill_executor: SkillExecutor | None = Field(
        default=None, description="Where skill scripts run (e.g. subprocess, E2B). None disables run_script."
    )
    input_schema: ClassVar[type[SkillsToolInputSchema]] = SkillsToolInputSchema

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {
            "skill_source": True,
            "skill_executor": True,
        }

    def execute(
        self, input_data: SkillsToolInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        action = input_data.action
        logger.info(f"SkillsTool: action '{action}'")

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
        if action == "run_script":
            if not self.skill_executor:
                raise ToolExecutionException(
                    "run_script is not available: skill_executor was not configured", recoverable=True
                )
            if not input_data.skill_name:
                raise ToolExecutionException("skill_name required for run_script", recoverable=True)
            if not input_data.script_path:
                raise ToolExecutionException(
                    "script_path required for run_script (e.g. scripts/run.py)", recoverable=True
                )
            return self._run_script(
                input_data.skill_name,
                input_data.script_path,
                input_data.arguments or [],
                input_files=input_data.input_files,
                output_paths=input_data.output_paths,
                output_prefix=input_data.output_prefix or "",
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
        logger.info(f"SkillsTool: listed {len(skills)} skills")
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
            logger.info(f"SkillsTool: got skill '{skill_name}' (section={section}, lines={line_start}-{line_end})")
            return {"content": out}
        skill = self.skill_source.load_skill(skill_name)
        if not skill:
            raise ToolExecutionException(f"Skill '{skill_name}' not found", recoverable=True)
        logger.info(f"SkillsTool: got skill '{skill_name}'")
        return {
            "content": {
                "skill_name": skill.name,
                "description": skill.metadata.description,
                "instructions": skill.get_full_content(),
                "supporting_files": [str(p) for p in skill.supporting_files_paths],
                "dependencies": skill.metadata.dependencies,
            }
        }

    def _run_script(
        self,
        skill_name: str,
        script_path: str,
        arguments: list[str],
        input_files: dict[str, str] | None = None,
        output_paths: list[str] | None = None,
        output_prefix: str = "",
    ) -> dict[str, Any]:
        if not isinstance(self.skill_executor, SkillExecutor):
            return {
                "content": {
                    "status": "error",
                    "message": "skill_executor is not a SkillExecutor instance",
                    "stdout": "",
                    "stderr": "",
                    "exit_code": -1,
                    "success": False,
                }
            }

        result = self.skill_executor.execute_script(
            skill_name=skill_name,
            script_relative_path=script_path.replace("\\", "/"),
            argv=arguments,
            input_files=input_files,
            output_paths=output_paths,
            output_prefix=output_prefix,
        )
        logger.info(f"SkillsTool: run_script {skill_name}/{script_path} {result.summary()}")

        content: dict[str, Any] = {
            "status": "completed" if result.success else "failed",
            "success": result.success,
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        if result.output_files:
            content["output_files"] = list(result.output_files.keys())
            files_list = []
            for path, data in result.output_files.items():
                bio = io.BytesIO(data)
                bio.name = path
                files_list.append(bio)
            return {"content": content, "files": files_list}
        return {"content": content}
