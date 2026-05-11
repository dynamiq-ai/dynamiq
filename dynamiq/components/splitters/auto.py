import enum
import json
import re
from pathlib import PurePath
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from dynamiq.components.splitters.base import IdStrategy, LengthUnit
from dynamiq.components.splitters.code import CodeParser, CodeSplitterComponent
from dynamiq.components.splitters.html import HTMLHeaderSplitterComponent, HTMLSectionSplitterComponent
from dynamiq.components.splitters.json import RecursiveJsonSplitterComponent
from dynamiq.components.splitters.language import Language
from dynamiq.components.splitters.markdown_header import MarkdownHeaderSplitterComponent
from dynamiq.components.splitters.recursive_character import RecursiveCharacterSplitterComponent
from dynamiq.types import Document
from dynamiq.utils.logger import logger


class AutoSplitterStrategy(str, enum.Enum):
    """Available splitter strategies for automatic document routing."""

    RECURSIVE_CHARACTER = "recursive_character"
    MARKDOWN_HEADER = "markdown_header"
    HTML_HEADER = "html_header"
    HTML_SECTION = "html_section"
    JSON = "json"
    CODE = "code"


class AutoSplitterRule(BaseModel):
    """Serializable rule used by :class:`AutoSplitterComponent` to select a splitter."""

    strategy: AutoSplitterStrategy
    file_types: list[str] = Field(default_factory=list)
    extensions: list[str] = Field(default_factory=list)
    content_types: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def normalize_rule_values(self) -> "AutoSplitterRule":
        self.file_types = [_normalize_token(value) for value in self.file_types]
        self.extensions = [_normalize_extension(value) for value in self.extensions]
        self.content_types = [_normalize_token(value) for value in self.content_types]
        self.metadata = {key: _normalize_token(value) for key, value in self.metadata.items()}
        return self


def _default_rules() -> list[AutoSplitterRule]:
    return [
        AutoSplitterRule(
            strategy=AutoSplitterStrategy.JSON,
            file_types=["json"],
            extensions=["json"],
            content_types=["application/json", "text/json"],
        ),
        AutoSplitterRule(
            strategy=AutoSplitterStrategy.CODE,
            extensions=[
                "c",
                "cc",
                "cpp",
                "cs",
                "go",
                "java",
                "js",
                "jsx",
                "kt",
                "kts",
                "lua",
                "php",
                "proto",
                "py",
                "rb",
                "rs",
                "scala",
                "sol",
                "swift",
                "ts",
                "tsx",
            ],
        ),
        AutoSplitterRule(
            strategy=AutoSplitterStrategy.HTML_SECTION,
            file_types=["html"],
            content_types=["text/html", "application/xhtml+xml"],
        ),
        AutoSplitterRule(
            strategy=AutoSplitterStrategy.MARKDOWN_HEADER,
            file_types=["markdown"],
            extensions=["md", "markdown", "mdx"],
            content_types=["text/markdown", "text/x-markdown"],
        ),
    ]


class AutoSplitterComponent(BaseModel):
    """Routes each input document to a structure-aware splitter when possible.

    The component expects already-converted :class:`Document` objects. It uses
    explicit metadata first, then file/content metadata, then lightweight content
    sniffing. Documents that cannot be classified use ``fallback_strategy``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    EXPLICIT_STRATEGY_METADATA_KEYS: ClassVar[tuple[str, ...]] = (
        "splitter_strategy",
        "splitter",
        "document_splitter",
    )

    chunk_size: int = Field(default=4000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    length_unit: LengthUnit = LengthUnit.CHARS
    keep_separator: bool = False
    strip_whitespace: bool = True
    add_start_index: bool = True
    add_chunk_index: bool = True
    merge_short_chunks: bool = True
    parent_chunk_size: int | None = None
    parent_chunk_overlap: int | None = None
    id_strategy: IdStrategy = IdStrategy.UUID

    rules: list[AutoSplitterRule] = Field(default_factory=_default_rules)
    fallback_strategy: AutoSplitterStrategy = AutoSplitterStrategy.RECURSIVE_CHARACTER
    fallback_on_error: bool = True
    infer_from_content: bool = True
    add_splitter_metadata: bool = True
    splitter_metadata_key: str = "splitter_strategy"

    json_max_chunk_size: int = Field(default=2000, gt=0)
    json_min_chunk_size: int | None = None
    json_convert_lists: bool = False

    markdown_headers_to_split_on: list[tuple[str, str]] | None = None
    markdown_strip_headers: bool = True
    markdown_return_each_line: bool = False

    html_headers_to_split_on: list[tuple[str, str]] | None = None
    html_return_each_element: bool = False
    html_xpath_filter: str | None = None

    code_parser: CodeParser = CodeParser.REGEX
    code_default_language: Language = Language.PYTHON

    _splitter_cache: dict[tuple[Any, ...], Any] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def validate_config(self) -> "AutoSplitterComponent":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")
        if self.parent_chunk_size is not None and self.parent_chunk_size < self.chunk_size:
            raise ValueError("parent_chunk_size must be >= chunk_size when set.")
        if self.parent_chunk_overlap is None:
            self.parent_chunk_overlap = self.chunk_overlap
        return self

    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        if not isinstance(documents, list):
            raise TypeError("AutoSplitter expects a list of Documents as input.")
        if any(not isinstance(document, Document) for document in documents):
            raise TypeError("AutoSplitter expects a list of Documents as input.")

        split_documents: list[Document] = []
        for document in documents:
            if document.content is None:
                raise ValueError(f"AutoSplitter requires text content; document ID {document.id} has none.")
            split_documents.extend(self._split_document(document))

        logger.debug(f"AutoSplitter: split {len(documents)} documents into {len(split_documents)} chunks.")
        return {"documents": split_documents}

    def _split_document(self, document: Document) -> list[Document]:
        strategy, explicit = self._select_strategy(document)
        try:
            output = self._get_splitter(strategy, document).run(documents=[document])
            chunks = output["documents"]
        except Exception as exc:
            if explicit or not self.fallback_on_error or strategy == self.fallback_strategy:
                raise
            logger.warning(
                "AutoSplitter strategy %s failed for document %s; falling back to %s. Error: %s",
                strategy.value,
                document.id,
                self.fallback_strategy.value,
                exc,
            )
            strategy = self.fallback_strategy
            chunks = self._get_splitter(strategy, document).run(documents=[document])["documents"]

        if self.add_splitter_metadata:
            for chunk in chunks:
                metadata = dict(chunk.metadata or {})
                metadata[self.splitter_metadata_key] = strategy.value
                chunk.metadata = metadata
        return chunks

    def _select_strategy(self, document: Document) -> tuple[AutoSplitterStrategy, bool]:
        metadata = document.metadata or {}
        explicit_strategy = self._explicit_strategy(metadata)
        if explicit_strategy is not None:
            return explicit_strategy, True

        features = _DocumentFeatures.from_document(document)
        content = document.content or ""

        if features.content_type:
            if "html" in features.content_type:
                return AutoSplitterStrategy.HTML_SECTION, False
            if "json" in features.content_type:
                return AutoSplitterStrategy.JSON, False
            if "markdown" in features.content_type:
                return AutoSplitterStrategy.MARKDOWN_HEADER, False

        if features.extension == "json":
            return AutoSplitterStrategy.JSON, False
        if features.extension in _EXTENSION_TO_LANGUAGE:
            return AutoSplitterStrategy.CODE, False

        if self.infer_from_content and _looks_like_html(content):
            return AutoSplitterStrategy.HTML_SECTION, False
        if self.infer_from_content and _looks_like_markdown(content):
            return AutoSplitterStrategy.MARKDOWN_HEADER, False
        if self.infer_from_content and _looks_like_json(content):
            return AutoSplitterStrategy.JSON, False

        for rule in self.rules:
            if features.matches(rule):
                return rule.strategy, False

        return self.fallback_strategy, False

    def _explicit_strategy(self, metadata: dict[str, Any]) -> AutoSplitterStrategy | None:
        for key in self.EXPLICIT_STRATEGY_METADATA_KEYS:
            if key not in metadata:
                continue
            strategy = _coerce_strategy(metadata[key])
            if strategy is None:
                raise ValueError(f"Unsupported splitter strategy in metadata['{key}']: {metadata[key]}")
            return strategy
        return None

    def _get_splitter(self, strategy: AutoSplitterStrategy, document: Document) -> Any:
        language = self._infer_code_language(document) if strategy == AutoSplitterStrategy.CODE else None
        cache_key = (strategy, language.value if language else None)
        if cache_key not in self._splitter_cache:
            self._splitter_cache[cache_key] = self._build_splitter(strategy, language)
        return self._splitter_cache[cache_key]

    def _build_splitter(self, strategy: AutoSplitterStrategy, language: Language | None = None) -> Any:
        if strategy == AutoSplitterStrategy.RECURSIVE_CHARACTER:
            return RecursiveCharacterSplitterComponent(**self._base_splitter_kwargs())
        if strategy == AutoSplitterStrategy.MARKDOWN_HEADER:
            return MarkdownHeaderSplitterComponent(
                headers_to_split_on=self.markdown_headers_to_split_on,
                strip_headers=self.markdown_strip_headers,
                return_each_line=self.markdown_return_each_line,
            )
        if strategy == AutoSplitterStrategy.HTML_HEADER:
            return HTMLHeaderSplitterComponent(
                headers_to_split_on=self.html_headers_to_split_on,
                return_each_element=self.html_return_each_element,
            )
        if strategy == AutoSplitterStrategy.HTML_SECTION:
            return HTMLSectionSplitterComponent(
                headers_to_split_on=self.html_headers_to_split_on,
                return_each_element=self.html_return_each_element,
                xpath_filter=self.html_xpath_filter,
            )
        if strategy == AutoSplitterStrategy.JSON:
            return RecursiveJsonSplitterComponent(
                max_chunk_size=self.json_max_chunk_size,
                min_chunk_size=self.json_min_chunk_size,
                convert_lists=self.json_convert_lists,
            )
        if strategy == AutoSplitterStrategy.CODE:
            return CodeSplitterComponent(
                language=language or self.code_default_language,
                parser=self.code_parser,
                keep_separator=True,
                **self._base_splitter_kwargs(exclude_keep_separator=True),
            )
        raise ValueError(f"Unsupported splitter strategy: {strategy}")

    def _base_splitter_kwargs(self, exclude_keep_separator: bool = False) -> dict[str, Any]:
        kwargs = {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "length_unit": self.length_unit,
            "strip_whitespace": self.strip_whitespace,
            "add_start_index": self.add_start_index,
            "add_chunk_index": self.add_chunk_index,
            "merge_short_chunks": self.merge_short_chunks,
            "parent_chunk_size": self.parent_chunk_size,
            "parent_chunk_overlap": self.parent_chunk_overlap,
            "id_strategy": self.id_strategy,
        }
        if not exclude_keep_separator:
            kwargs["keep_separator"] = self.keep_separator
        return kwargs

    def _infer_code_language(self, document: Document) -> Language:
        features = _DocumentFeatures.from_document(document)
        if features.extension in _EXTENSION_TO_LANGUAGE:
            return _EXTENSION_TO_LANGUAGE[features.extension]
        language = (document.metadata or {}).get("language")
        if language:
            try:
                return Language(str(language).lower())
            except ValueError:
                logger.debug("Unsupported code language metadata value: %s", language)
        return self.code_default_language


class _DocumentFeatures(BaseModel):
    file_type: str | None = None
    extension: str | None = None
    content_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_document(cls, document: Document) -> "_DocumentFeatures":
        metadata = document.metadata or {}
        file_path = metadata.get("file_path") or metadata.get("source") or metadata.get("filename")
        extension = metadata.get("extension") or metadata.get("file_extension")
        if not extension and file_path:
            suffix = PurePath(str(file_path)).suffix
            extension = suffix[1:] if suffix.startswith(".") else suffix

        file_type = metadata.get("file_type") or metadata.get("document_type") or metadata.get("type")
        content_type = metadata.get("content_type") or metadata.get("mime_type") or metadata.get("mimetype")

        return cls(
            file_type=_normalize_token(file_type) if file_type else None,
            extension=_normalize_extension(extension) if extension else None,
            content_type=_normalize_token(content_type) if content_type else None,
            metadata=metadata,
        )

    def matches(self, rule: AutoSplitterRule) -> bool:
        if rule.file_types and self.file_type in rule.file_types:
            return True
        if rule.extensions and self.extension in rule.extensions:
            return True
        if rule.content_types and self.content_type:
            if any(
                self.content_type == value or self.content_type.startswith(f"{value};") for value in rule.content_types
            ):
                return True
        for key, expected in rule.metadata.items():
            actual = self.metadata.get(key)
            if actual is None or _normalize_token(actual) != expected:
                return False
        return bool(rule.metadata)


_STRATEGY_ALIASES = {
    "recursive": AutoSplitterStrategy.RECURSIVE_CHARACTER,
    "recursive_character": AutoSplitterStrategy.RECURSIVE_CHARACTER,
    "recursivecharactersplitter": AutoSplitterStrategy.RECURSIVE_CHARACTER,
    "markdown": AutoSplitterStrategy.MARKDOWN_HEADER,
    "markdown_header": AutoSplitterStrategy.MARKDOWN_HEADER,
    "markdownheadersplitter": AutoSplitterStrategy.MARKDOWN_HEADER,
    "html": AutoSplitterStrategy.HTML_SECTION,
    "html_header": AutoSplitterStrategy.HTML_HEADER,
    "htmlheadersplitter": AutoSplitterStrategy.HTML_HEADER,
    "html_section": AutoSplitterStrategy.HTML_SECTION,
    "htmlsectionsplitter": AutoSplitterStrategy.HTML_SECTION,
    "json": AutoSplitterStrategy.JSON,
    "recursive_json": AutoSplitterStrategy.JSON,
    "recursivejsonsplitter": AutoSplitterStrategy.JSON,
    "code": AutoSplitterStrategy.CODE,
    "codesplitter": AutoSplitterStrategy.CODE,
}

_EXTENSION_TO_LANGUAGE = {
    "c": Language.C,
    "cc": Language.CPP,
    "cpp": Language.CPP,
    "cs": Language.CSHARP,
    "go": Language.GO,
    "java": Language.JAVA,
    "js": Language.JS,
    "jsx": Language.JS,
    "kt": Language.KOTLIN,
    "kts": Language.KOTLIN,
    "lua": Language.LUA,
    "php": Language.PHP,
    "proto": Language.PROTO,
    "py": Language.PYTHON,
    "rb": Language.RUBY,
    "rs": Language.RUST,
    "scala": Language.SCALA,
    "sol": Language.SOL,
    "swift": Language.SWIFT,
    "ts": Language.TS,
    "tsx": Language.TS,
}


def _coerce_strategy(value: Any) -> AutoSplitterStrategy | None:
    if isinstance(value, AutoSplitterStrategy):
        return value
    normalized = _normalize_token(value).replace("-", "_").replace(" ", "_")
    try:
        return AutoSplitterStrategy(normalized)
    except ValueError:
        return _STRATEGY_ALIASES.get(normalized)


def _normalize_token(value: Any) -> str:
    return str(value).strip().lower()


def _normalize_extension(value: Any) -> str:
    return _normalize_token(value).lstrip(".")


def _looks_like_html(text: str) -> bool:
    return (
        re.search(
            r"<\s*(?:!doctype\s+html|html|body|h[1-6]|p|div|section|article|table|ul|ol|pre|code)\b",
            text[:4000],
            re.IGNORECASE,
        )
        is not None
    )


def _looks_like_markdown(text: str) -> bool:
    return re.search(r"(?m)^\s{0,3}#{1,6}\s+\S", text[:4000]) is not None


def _looks_like_json(text: str) -> bool:
    stripped = text.strip()
    if not stripped or stripped[0] not in "[{":
        return False
    try:
        json.loads(stripped)
    except json.JSONDecodeError:
        return False
    return True
