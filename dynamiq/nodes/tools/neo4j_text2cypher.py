import json
import re
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from dynamiq.nodes import ErrorHandling, Node, NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import NodeDependency, ensure_config
from dynamiq.prompts import Message, Prompt
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.utils.logger import logger

PROMPT_TEMPLATE_TEXT2CYPHER = """
You are a senior Neo4j Cypher expert. Given a user question and a database schema description, generate ONE safe, parameterized Cypher query.

ALWAYS follow these rules:
- Respond ONLY with compact JSON on a single line: {{"cypher": "<query>", "params": {{}}, "reasoning": "<short>"}}.
- Use parameters for dynamic values (e.g., $name, $limit) and include them in params.
- Prefer MATCH/OPTIONAL MATCH/WHERE/RETURN for reads. Use LIMIT {default_limit} unless the question specifies otherwise.
- Do NOT invent labels or relationship types; stay within the provided schema/context. If schema is missing, make minimal assumptions and document them in reasoning.
- If writes are not allowed: avoid CREATE, MERGE, SET, DELETE, REMOVE, DROP, CALL dbms components that mutate state.
- Always return meaningful fields (ids, names, properties) and avoid returning entire nodes unless required.
- If unsure, return an empty cypher string and explain briefly in reasoning.

User question:
{question}

Schema/context (may include labels, relationships, properties, example patterns):
{schema}

Additional examples or hints (optional):
{examples}

Write allowance: {write_policy}
"""  # noqa: E501


class Neo4jText2CypherInputSchema(BaseModel):
    question: str = Field(..., description="Natural language question to translate into Cypher.")
    schema_text: str = Field(
        default="",
        description="Schema/context text describing labels, relationships, properties, or example Cypher patterns.",
    )
    examples: list[str] = Field(
        default_factory=list,
        description="Optional example Cypher snippets or QA pairs to guide generation.",
    )
    allow_writes: bool = Field(default=False, description="If true, writes are permitted (CREATE/MERGE/SET/DELETE).")
    default_limit: int = Field(default=25, description="Default LIMIT to apply when not specified.")

    @model_validator(mode="after")
    def validate_limit(self):
        if self.default_limit <= 0:
            raise ValueError("default_limit must be positive.")
        return self

    model_config = ConfigDict(extra="forbid")


class Neo4jText2Cypher(Node):
    """LLM-backed tool that generates guarded Cypher from natural language."""

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "Neo4j Text2Cypher"
    description: str = (
        "Generates safe, parameterized Cypher for Neo4j from natural language with schema-aware guardrails. "
        "Inputs: question, schema_text (labels/relationships/properties/examples), "
        "examples (list), allow_writes (bool), default_limit (int). "
        "Outputs: cypher, params, reasoning, content summary."
    )
    error_handling: ErrorHandling = Field(default_factory=lambda: ErrorHandling(timeout_seconds=600))
    llm: Node

    input_schema: ClassVar[type[Neo4jText2CypherInputSchema]] = Neo4jText2CypherInputSchema
    _run_depends: list[dict] = PrivateAttr(default_factory=list)

    def init_components(self, connection_manager=None):
        connection_manager = connection_manager or None
        super().init_components(connection_manager)
        if self.llm.is_postponed_component_init:
            self.llm.init_components(connection_manager)

    def reset_run_state(self):
        self._run_depends = []

    @property
    def to_dict_exclude_params(self) -> dict:
        return super().to_dict_exclude_params | {"llm": True}

    def to_dict(self, **kwargs) -> dict:
        data = super().to_dict(**kwargs)
        data["llm"] = self.llm.to_dict(**kwargs)
        return data

    def execute(self, input_data: Neo4jText2CypherInputSchema, config: RunnableConfig = None, **kwargs):
        logger.info(f"Tool {self.name} - {self.id}: started with INPUT DATA:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.reset_run_state()
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        prompt = self._build_prompt(input_data)

        result = self.llm.run(
            input_data={},
            prompt=Prompt(messages=[Message(role="user", content=prompt)]),
            config=config,
            **(kwargs | {"parent_run_id": kwargs.get("run_id")}),
        )

        self._run_depends = [NodeDependency(node=self.llm).to_dict(for_tracing=True)]

        if result.status != RunnableStatus.SUCCESS:
            raise ToolExecutionException("LLM execution failed while generating Cypher.", recoverable=True)

        raw_output = result.output.get("content", "")
        parsed = self._parse_llm_output(raw_output)

        if not input_data.allow_writes and self._contains_write(parsed.get("cypher", "")):
            raise ToolExecutionException("Generated Cypher contains write operations but writes are disabled.")

        cypher = parsed.get("cypher", "") or ""
        params = parsed.get("params", {}) or {}
        reasoning = parsed.get("reasoning", "") or ""
        if not cypher.strip():
            raise ToolExecutionException(
                "Generated Cypher is empty; please refine the question or schema.", recoverable=True
            )

        parsed["content"] = f"Cypher: {cypher}\nParams: {params}\nReasoning: {reasoning}"

        logger.info(f"Tool {self.name} - {self.id}: finished successfully. Output: {parsed}")
        return parsed

    def _build_prompt(self, input_data: Neo4jText2CypherInputSchema) -> str:
        examples_block = "\n".join(input_data.examples) if input_data.examples else "None provided."
        write_policy = "Writes allowed (use CREATE/MERGE when needed)." if input_data.allow_writes else "Read-only."
        return PROMPT_TEMPLATE_TEXT2CYPHER.format(
            question=input_data.question.strip(),
            schema=input_data.schema_text.strip() or "Not provided.",
            examples=examples_block,
            write_policy=write_policy,
            default_limit=input_data.default_limit,
        )

    @staticmethod
    def _parse_llm_output(text: str) -> dict[str, Any]:
        try:
            cleaned = text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`").strip()
                if cleaned.startswith("json"):
                    cleaned = cleaned[len("json") :].strip()
            return json.loads(cleaned)
        except Exception as exc:  # noqa: BLE001
            try:
                match = re.search(r"\{.*\}", text, flags=re.DOTALL)
                if match:
                    return json.loads(match.group(0))
            except Exception as fallback_exc:  # noqa: BLE001
                logger.debug(
                    "Failed to parse JSON from LLM output; primary error=%s, fallback error=%s, text=%s",
                    exc,
                    fallback_exc,
                    text,
                )
        raise ToolExecutionException(
            f"Failed to parse LLM output into JSON from text: {text!r}",
            recoverable=True,
        )

    @staticmethod
    def _contains_write(cypher: str) -> bool:
        pattern = re.compile(r"\b(CREATE|MERGE|DELETE|DETACH|SET|DROP)\b", re.IGNORECASE)
        return bool(pattern.search(cypher or ""))
