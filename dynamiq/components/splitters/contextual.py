import hashlib
from copy import deepcopy
from typing import Any, Callable

from dynamiq.types import Document
from dynamiq.utils.logger import logger

DEFAULT_CONTEXT_PROMPT = (
    "<document>\n{document}\n</document>\n"
    "Here is the chunk we want to situate within the whole document:\n"
    "<chunk>\n{chunk}\n</chunk>\n"
    "Please give a short succinct context (1-3 sentences) to situate this chunk within "
    "the overall document for the purposes of improving search retrieval of the chunk. "
    "Answer only with the succinct context and nothing else."
)


class ContextualChunkerComponent:
    """Wraps an inner splitter and prepends LLM-generated doc-level context to each chunk.

    ``inner_splitter`` must expose ``run(documents) -> {"documents": list[Document]}``.
    ``llm_fn`` accepts a fully-formatted prompt string and returns the model response.
    """

    METADATA_KEY = "context"

    def __init__(
        self,
        inner_splitter: Any,
        llm_fn: Callable[[str], str],
        context_prompt: str = DEFAULT_CONTEXT_PROMPT,
        prepend: bool = True,
        cache_context: bool = True,
        separator: str = "\n\n",
    ) -> None:
        if not (hasattr(inner_splitter, "run") or hasattr(inner_splitter, "execute")):
            raise TypeError("inner_splitter must expose a run(documents) or execute(input_data) method.")
        self.inner_splitter = inner_splitter
        self.llm_fn = llm_fn
        self.context_prompt = context_prompt
        self.prepend = prepend
        self.cache_context = cache_context
        self.separator = separator
        self._cache: dict[str, str] = {}

    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        if not isinstance(documents, list):
            raise TypeError("ContextualChunker expects a list of Documents as input.")
        outputs: list[Document] = []
        for doc in documents:
            if doc.content is None:
                raise ValueError(f"ContextualChunker requires text content; document ID {doc.id} has none.")
            split_result = self._run_inner_splitter([doc])
            child_chunks: list[Document] = split_result["documents"]
            for chunk in child_chunks:
                context = self._get_context(doc.content, chunk.content)
                metadata = deepcopy(chunk.metadata) if chunk.metadata else {}
                metadata[self.METADATA_KEY] = context
                content = f"{context}{self.separator}{chunk.content}" if self.prepend else chunk.content
                outputs.append(Document(id=chunk.id, content=content, metadata=metadata, embedding=chunk.embedding))
        logger.debug(f"ContextualChunker: produced {len(outputs)} chunks from {len(documents)} documents.")
        return {"documents": outputs}

    def _run_inner_splitter(self, documents: list[Document]) -> dict[str, list[Document]]:
        input_schema = getattr(self.inner_splitter, "input_schema", None)
        if input_schema is not None and hasattr(self.inner_splitter, "execute"):
            return self.inner_splitter.execute(input_schema(documents=documents))
        return self.inner_splitter.run(documents=documents)

    def _get_context(self, document_text: str, chunk_text: str) -> str:
        cache_key = hashlib.sha256(f"{document_text}:{chunk_text}".encode()).hexdigest() if self.cache_context else ""
        if self.cache_context and cache_key in self._cache:
            return self._cache[cache_key]
        prompt = self.context_prompt.format(document=document_text, chunk=chunk_text)
        context = self.llm_fn(prompt).strip()
        if self.cache_context:
            self._cache[cache_key] = context
        return context
