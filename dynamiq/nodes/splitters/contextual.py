from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.components.splitters.contextual import DEFAULT_CONTEXT_PROMPT, ContextualChunkerComponent
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.llms.base import BaseLLM, BaseLLMInputSchema
from dynamiq.nodes.node import Node, NodeGroup, ensure_config
from dynamiq.prompts.prompts import Message, MessageRole, Prompt
from dynamiq.runnables import RunnableConfig
from dynamiq.types import Document
from dynamiq.utils.logger import logger


class ContextualChunkerInputSchema(BaseModel):
    documents: list[Document] = Field(..., description="Documents to split with contextual prepend.")


class ContextualChunker(Node):
    """Anthropic-style Contextual Retrieval splitter.

    Wraps an inner splitter node, then for each chunk asks an LLM to produce a
    short doc-level context which is prepended (or stored as metadata).
    """

    group: Literal[NodeGroup.SPLITTERS] = NodeGroup.SPLITTERS
    name: str = "ContextualChunker"
    description: str = "Wraps an inner splitter and prepends LLM-generated doc-level context to each chunk."

    inner_splitter: Node | None = Field(default=None, description="Splitter node used to produce raw chunks.")
    llm: BaseLLM | None = Field(default=None, description="LLM node used to generate the contextual sentence(s).")
    context_prompt: str = Field(
        default=DEFAULT_CONTEXT_PROMPT, description="Prompt template with `{document}` and `{chunk}` placeholders."
    )
    prepend: bool = Field(default=True, description="Prepend context to chunk content (else only stored in metadata).")
    cache_context: bool = Field(default=True, description="Cache LLM responses by (document, chunk) hash.")
    separator: str = Field(default="\n\n", description="String inserted between context and chunk content.")

    splitter: Any | None = None
    input_schema: ClassVar[type[ContextualChunkerInputSchema]] = ContextualChunkerInputSchema

    @property
    def to_dict_exclude_params(self) -> dict[str, Any]:
        return super().to_dict_exclude_params | {"splitter": True, "inner_splitter": True, "llm": True}

    def to_dict(self, **kwargs) -> dict[str, Any]:
        data = super().to_dict(**kwargs)
        if self.inner_splitter is not None:
            data["inner_splitter"] = self.inner_splitter.to_dict(**kwargs)
        if self.llm is not None:
            data["llm"] = self.llm.to_dict(**kwargs)
        return data

    def init_components(self, connection_manager: ConnectionManager | None = None) -> None:
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        if self.inner_splitter is None:
            raise ValueError("ContextualChunker requires an `inner_splitter` node.")
        if self.llm is None:
            raise ValueError("ContextualChunker requires an `llm` node.")
        self.inner_splitter.init_components(connection_manager)
        self.llm.init_components(connection_manager)
        if self.splitter is None:
            self.splitter = ContextualChunkerComponent(
                inner_splitter=self.inner_splitter,
                llm_fn=self._call_llm,
                context_prompt=self.context_prompt,
                prepend=self.prepend,
                cache_context=self.cache_context,
                separator=self.separator,
            )

    def _call_llm(self, prompt_text: str) -> str:
        prompt = Prompt(messages=[Message(role=MessageRole.USER, content=prompt_text)])
        input_data = BaseLLMInputSchema.model_validate({}, context={"prompt": prompt})
        result = self.llm.execute(input_data=input_data, prompt=prompt)
        return result.get("content") or ""

    def execute(
        self, input_data: ContextualChunkerInputSchema, config: RunnableConfig = None, **kwargs
    ) -> dict[str, Any]:
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)
        documents = input_data.documents
        logger.debug(f"ContextualChunker: splitting {len(documents)} documents.")
        output = self.splitter.run(documents=documents)
        return {"documents": output["documents"]}

    def run_with_inner_splitter(self, documents: list[Document]) -> dict[str, Any]:
        return self.splitter.run(documents=documents)
