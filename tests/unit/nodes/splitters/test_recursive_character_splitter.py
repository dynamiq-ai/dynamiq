from dynamiq.components.splitters.language import Language
from dynamiq.nodes.splitters.recursive_character import RecursiveCharacterSplitter
from dynamiq.types import Document


def test_recursive_character_basic_split():
    splitter = RecursiveCharacterSplitter(chunk_size=30, chunk_overlap=5)
    splitter.init_components()
    text = "Sentence one. Sentence two. Sentence three.\n\nNext paragraph here for splitting."
    output = splitter.execute(splitter.input_schema(documents=[Document(content=text)]))
    chunks = output["documents"]
    assert len(chunks) > 1
    assert all(len(chunk.content) <= 60 for chunk in chunks)
    assert all("source_id" in chunk.metadata for chunk in chunks)
    assert chunks[0].metadata["chunk_index"] == 0


def test_recursive_character_metadata_propagation():
    splitter = RecursiveCharacterSplitter(chunk_size=20, chunk_overlap=0)
    splitter.init_components()
    document = Document(content="alpha beta gamma delta epsilon zeta", metadata={"origin": "unit-test"})
    chunks = splitter.execute(splitter.input_schema(documents=[document]))["documents"]
    assert all(chunk.metadata["origin"] == "unit-test" for chunk in chunks)
    assert all(chunk.metadata["source_id"] == document.id for chunk in chunks)


def test_recursive_character_language_preset_replaces_separators():
    splitter = RecursiveCharacterSplitter(chunk_size=80, chunk_overlap=0, language=Language.PYTHON)
    splitter.init_components()
    code = "def foo():\n    return 1\n\ndef bar():\n    return 2"
    chunks = splitter.execute(splitter.input_schema(documents=[Document(content=code)]))["documents"]
    assert len(chunks) >= 1
    assert any("def foo" in chunk.content for chunk in chunks)


def test_recursive_character_to_dict_excludes_runtime_splitter():
    splitter = RecursiveCharacterSplitter(chunk_size=50, chunk_overlap=5)
    splitter.init_components()
    assert "splitter" in splitter.to_dict_exclude_params
    serialized = splitter.model_dump(exclude={"splitter"})
    assert serialized["chunk_size"] == 50
