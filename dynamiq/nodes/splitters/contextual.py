from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.components.splitters.contextual import DEFAULT_CONTEXT_PROMPT, ContextualSplitterComponent
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.llms.base import BaseLLM
from dynamiq.nodes.node import Node, NodeDependency, NodeGroup, ensure_config
from dynamiq.prompts.prompts import Message, MessageRole, Prompt
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types import Document
from dynamiq.utils.logger import logger


class ContextualSplitterInputSchema(BaseModel):
    documents: list[Document] = Field(..., description="Documents to split with contextual prepend.")


class ContextualSplitter(Node):
    """Anthropic-style Contextual Retrieval splitter.

    Wraps an inner splitter node, then for each split asks an LLM to produce a
    short doc-level context which is prepended (or stored as metadata).
    """

    group: Literal[NodeGroup.SPLITTERS] = NodeGroup.SPLITTERS
    name: str = "ContextualSplitter"
    description: str = "Prepends LLM-generated document context to splitter output."

    inner_splitter: Node | None = Field(default=None, description="Splitter node used to produce raw splits.")
    llm: BaseLLM | None = Field(
        default=None,
        description="LLM node used to generate the contextual sentence(s).",
    )
    context_prompt: str = Field(
        default=DEFAULT_CONTEXT_PROMPT,
        description="Prompt template with `{document}` and `{chunk}` placeholders.",
    )
    prepend: bool = Field(
        default=True,
        description="Prepend context to split content (else only stored in metadata).",
    )
    cache_context: bool = Field(default=True, description="Cache LLM responses by (document, chunk) hash.")
    separator: str = Field(default="\n\n", description="String inserted between context and chunk content.")

    splitter: Any | None = None
    input_schema: ClassVar[type[BaseModel]] = ContextualSplitterInputSchema

    @property
    def to_dict_exclude_params(self) -> dict[str, Any]:
        return super().to_dict_exclude_params | {
            "splitter": True,
            "inner_splitter": True,
            "llm": True,
        }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._run_depends = []

    def reset_run_state(self) -> None:
        self._run_depends = []

    def to_dict(
        self,
        include_secure_params: bool = False,
        for_tracing: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        data = super().to_dict(
            include_secure_params=include_secure_params,
            for_tracing=for_tracing,
            **kwargs,
        )
        if self.inner_splitter is not None:
            data["inner_splitter"] = self.inner_splitter.to_dict(
                include_secure_params=include_secure_params,
                for_tracing=for_tracing,
                **kwargs,
            )
        if self.llm is not None:
            data["llm"] = self.llm.to_dict(
                include_secure_params=include_secure_params,
                for_tracing=for_tracing,
                **kwargs,
            )
        return data

    def init_components(self, connection_manager: ConnectionManager | None = None) -> None:
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        if self.inner_splitter is None:
            raise ValueError("ContextualSplitter requires an `inner_splitter` node.")
        if self.llm is None:
            raise ValueError("ContextualSplitter requires an `llm` node.")
        self.inner_splitter.init_components(connection_manager)
        self.llm.init_components(connection_manager)
        if self.splitter is None:
            self.splitter = ContextualSplitterComponent(
                inner_splitter=self.inner_splitter,
                llm_fn=self._call_llm,
                context_prompt=self.context_prompt,
                prepend=self.prepend,
                cache_context=self.cache_context,
                separator=self.separator,
            )

    def _call_llm(
        self,
        prompt_text: str,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> str:
        config = ensure_config(config)
        prompt = Prompt(messages=[Message(role=MessageRole.USER, content=prompt_text)])
        result = self.llm.run(
            input_data={},
            prompt=prompt,
            config=config,
            run_depends=getattr(self, "_run_depends", []),
            **(kwargs | {"parent_run_id": kwargs.get("run_id")}),
        )
        if isinstance(self.llm, Node):
            self._run_depends = [NodeDependency(node=self.llm).to_dict(for_tracing=True)]
        if result.status != RunnableStatus.SUCCESS:
            raise ValueError("ContextualSplitter LLM execution failed")
        return result.output.get("content") or ""

    def execute(
        self,
        input_data: ContextualSplitterInputSchema,
        config: RunnableConfig = None,
        **kwargs,
    ) -> dict[str, Any]:
        config = ensure_config(config)
        self.reset_run_state()
        self.run_on_node_execute_run(config.callbacks, **kwargs)
        documents = input_data.documents
        logger.debug(f"ContextualSplitter: splitting {len(documents)} documents.")
        output = self.splitter.run(documents=documents, config=config, **kwargs)
        return {"documents": output["documents"]}

    def run_with_inner_splitter(self, documents: list[Document]) -> dict[str, Any]:
        return self.splitter.run(documents=documents)
