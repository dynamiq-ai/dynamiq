from dynamiq.components.splitters.code import CodeParser, CodeSplitterComponent
from dynamiq.components.splitters.language import Language
from dynamiq.nodes.splitters.code import CodeSplitter
from dynamiq.types import Document


def test_code_splitter_python_breaks_on_def():
    splitter = CodeSplitter(language=Language.PYTHON, chunk_size=80, chunk_overlap=0)
    splitter.init_components()
    code = (
        "def foo():\n    return 1\n\n"
        "def bar():\n    return 2\n\n"
        "class Baz:\n    def method(self):\n        return 3\n"
    )
    chunks = splitter.execute(splitter.input_schema(documents=[Document(content=code)]))["documents"]
    assert len(chunks) >= 2
    assert any("def foo" in chunk.content for chunk in chunks)
    assert any("class Baz" in chunk.content for chunk in chunks)


def test_code_splitter_excludes_runtime_splitter_from_serialization():
    splitter = CodeSplitter(language=Language.JS, chunk_size=100, chunk_overlap=10)
    splitter.init_components()
    assert "splitter" in splitter.to_dict_exclude_params
    serialized = splitter.model_dump(exclude={"splitter"})
    assert serialized["language"] == Language.JS
    assert serialized["chunk_size"] == 100


def test_code_parent_splitter_preserves_language_and_parser_settings():
    splitter = CodeSplitterComponent(
        language=Language.JS,
        parser=CodeParser.TREE_SITTER,
        chunk_size=40,
        chunk_overlap=0,
        parent_chunk_size=120,
    )

    parent_kwargs = splitter._parent_splitter_kwargs()

    assert parent_kwargs["language"] == Language.JS
    assert parent_kwargs["parser"] == CodeParser.TREE_SITTER
    assert parent_kwargs["separators"][0] == "\nfunction "
    assert parent_kwargs["is_separator_regex"] is True
    assert parent_kwargs["add_start_index"] is True
