import json

from dynamiq.nodes.splitters.json import RecursiveJsonSplitter
from dynamiq.types import Document


def test_recursive_json_splitter_respects_max_chunk_size():
    splitter = RecursiveJsonSplitter(max_chunk_size=50)
    splitter.init_components()
    payload = {f"key_{i}": "x" * 10 for i in range(10)}
    document = Document(content=json.dumps(payload))
    chunks = splitter.execute(splitter.input_schema(documents=[document]))["documents"]
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.content) <= 60  # small slack for JSON braces


def test_recursive_json_splitter_handles_nested_lists():
    splitter = RecursiveJsonSplitter(max_chunk_size=80, convert_lists=True)
    splitter.init_components()
    payload = {"items": [{"name": "a" * 30}, {"name": "b" * 30}, {"name": "c" * 30}]}
    document = Document(content=json.dumps(payload))
    chunks = splitter.execute(splitter.input_schema(documents=[document]))["documents"]
    assert len(chunks) > 1
    for chunk in chunks:
        json.loads(chunk.content)
