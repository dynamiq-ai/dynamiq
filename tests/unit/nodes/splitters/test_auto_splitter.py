import json

from dynamiq import Workflow
from dynamiq.components.splitters.auto import (
    AutoSplitterRule,
    AutoSplitterStrategy,
    _default_rules,
    _repair_flattened_markdown,
)
from dynamiq.flows import Flow
from dynamiq.nodes.converters import TextFileConverter
from dynamiq.nodes.node import NodeDependency
from dynamiq.nodes.splitters import AutoSplitter
from dynamiq.runnables import RunnableStatus
from dynamiq.types import Document


def test_auto_splitter_routes_markdown_by_extension():
    splitter = AutoSplitter()
    splitter.init_components()
    document = Document(
        content="# Title\nIntro\n## Details\nBody",
        metadata={"file_path": "notes.md"},
    )

    chunks = splitter.execute(splitter.input_schema(documents=[document]))["documents"]

    assert len(chunks) == 2
    assert chunks[0].metadata["h1"] == "Title"
    assert chunks[1].metadata["h2"] == "Details"
    assert all(chunk.metadata["source_id"] == document.id for chunk in chunks)
    assert all(chunk.metadata["splitter_strategy"] == "markdown_header" for chunk in chunks)


def test_auto_splitter_repairs_flattened_markdown_headings_before_splitting():
    content = "Navigation text # Article title Answer text ## Details More information"
    splitter = AutoSplitter(markdown_strip_headers=False)
    splitter.init_components()
    document = Document(content=content, metadata={"file_path": "article.md"})

    chunks = splitter.execute(splitter.input_schema(documents=[document]))["documents"]

    assert [chunk.content for chunk in chunks] == [
        "Navigation text",
        "# Article title Answer text",
        "## Details More information",
    ]
    assert chunks[1].metadata["h1"] == "Article title Answer text"
    assert chunks[2].metadata["h2"] == "Details More information"
    assert all(chunk.metadata["content_normalization"] == "flattened_markdown" for chunk in chunks)
    assert all(chunk.metadata["content_normalization_length_preserving"] is True for chunk in chunks)


def test_flattened_markdown_repair_is_length_preserving_and_skips_structured_documents():
    flattened = "Introduction #### Section Body ##### Feature detail"
    repaired = _repair_flattened_markdown(flattened)

    assert repaired == "Introduction\n#### Section Body ##### Feature detail"
    assert len(repaired) == len(flattened)

    structured = "# First\nBody\nInline text ## not a heading"
    assert _repair_flattened_markdown(structured) == structured


def test_auto_splitter_infers_flattened_markdown_without_file_metadata():
    splitter = AutoSplitter(markdown_strip_headers=False)
    splitter.init_components()
    document = Document(content="Preamble ## First section Body ### Second section More")

    chunks = splitter.execute(splitter.input_schema(documents=[document]))["documents"]

    assert len(chunks) == 3
    assert all(chunk.metadata["splitter_strategy"] == "markdown_header" for chunk in chunks)
    assert all(chunk.metadata["content_normalization"] == "flattened_markdown" for chunk in chunks)


def test_auto_splitter_routes_from_converter_file_type_without_filename_or_content_inference():
    splitter = AutoSplitter(infer_from_content=False)
    splitter.init_components()
    document = Document(
        content="# Title\nIntro\n## Details\nBody",
        metadata={"file_type": "markdown", "source": "item-id-without-extension"},
    )

    chunks = splitter.execute(splitter.input_schema(documents=[document]))["documents"]

    assert len(chunks) == 2
    assert all(chunk.metadata["splitter_strategy"] == "markdown_header" for chunk in chunks)


def test_auto_splitter_does_not_infer_markdown_for_stamped_csv_content():
    content = "Plan: Basic # Features support ## Pricing free"
    splitter = AutoSplitter(markdown_strip_headers=False)
    splitter.init_components()
    document = Document(content=content, metadata={"file_type": "csv", "source": "upload-id"})

    chunks = splitter.execute(splitter.input_schema(documents=[document]))["documents"]

    assert [chunk.content for chunk in chunks] == [content]
    assert all(chunk.metadata["splitter_strategy"] == "recursive_character" for chunk in chunks)
    assert all("content_normalization" not in chunk.metadata for chunk in chunks)


def test_auto_splitter_allows_explicit_rules_for_stamped_plain_content():
    splitter = AutoSplitter(rules=[AutoSplitterRule(strategy=AutoSplitterStrategy.MARKDOWN_HEADER, file_types=["csv"])])
    splitter.init_components()
    document = Document(content="# Title\nBody", metadata={"file_type": "csv"})

    chunks = splitter.execute(splitter.input_schema(documents=[document]))["documents"]

    assert chunks[0].metadata["splitter_strategy"] == "markdown_header"


def test_auto_splitter_refines_oversized_markdown_sections_and_preserves_source_metadata():
    splitter = AutoSplitter(chunk_size=45, chunk_overlap=0)
    splitter.init_components()
    document = Document(
        content="# Pricing\n" + " ".join(["enterprise-plan"] * 12),
        metadata={
            "file_path": "pricing.md",
            "dynamiq_item_source_provider_url": "https://example.com/pricing",
        },
    )

    chunks = splitter.execute(splitter.input_schema(documents=[document]))["documents"]

    assert len(chunks) > 1
    assert all(len(chunk.content) <= 45 for chunk in chunks)
    assert all(chunk.metadata["h1"] == "Pricing" for chunk in chunks)
    assert all(chunk.metadata["source_id"] == document.id for chunk in chunks)
    assert all(chunk.metadata["dynamiq_item_source_provider_url"] == "https://example.com/pricing" for chunk in chunks)
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(range(len(chunks)))


def test_auto_splitter_maps_repeated_structured_sections_to_monotonic_source_offsets():
    repeated_body = " ".join(["same-section-content"] * 8)
    source_text = f"# First\n{repeated_body}\n# Second\n{repeated_body}"
    splitter = AutoSplitter(chunk_size=45, chunk_overlap=0)
    splitter.init_components()

    chunks = splitter.execute(
        splitter.input_schema(documents=[Document(content=source_text, metadata={"file_type": "markdown"})])
    )["documents"]

    first_offsets = [chunk.metadata["start_index"] for chunk in chunks if chunk.metadata.get("h1") == "First"]
    second_offsets = [chunk.metadata["start_index"] for chunk in chunks if chunk.metadata.get("h1") == "Second"]
    assert first_offsets and second_offsets
    assert max(first_offsets) < min(second_offsets)
    assert min(second_offsets) >= source_text.index(repeated_body, source_text.index("# Second"))


def test_auto_splitter_maps_multi_paragraph_markdown_chunks_to_source_offsets():
    first_body = " ".join(f"first-{index}" for index in range(16))
    first_tail = " ".join(f"first-tail-{index}" for index in range(10))
    second_body = " ".join(f"second-{index}" for index in range(16))
    second_tail = " ".join(f"second-tail-{index}" for index in range(10))
    source_text = "\n".join(
        [
            f"# First\n{first_body}\n\n{first_tail}",
            f"# Second\n{second_body}\n\n{second_tail}",
        ]
    )
    splitter = AutoSplitter(chunk_size=45, chunk_overlap=0)
    splitter.init_components()

    chunks = splitter.execute(
        splitter.input_schema(documents=[Document(content=source_text, metadata={"file_type": "markdown"})])
    )["documents"]

    assert len(chunks) > 2
    assert {chunk.metadata.get("h1") for chunk in chunks} == {"First", "Second"}
    assert all(chunk.metadata["start_index"] == source_text.index(chunk.content) for chunk in chunks)


def test_auto_splitter_routes_json_by_extension():
    splitter = AutoSplitter(json_max_chunk_size=50)
    splitter.init_components()
    payload = {f"key_{i}": "x" * 10 for i in range(8)}
    document = Document(content=json.dumps(payload), metadata={"file_path": "payload.json"})

    chunks = splitter.execute(splitter.input_schema(documents=[document]))["documents"]

    assert len(chunks) > 1
    assert all(chunk.metadata["splitter_strategy"] == "json" for chunk in chunks)
    for chunk in chunks:
        json.loads(chunk.content)


def test_auto_splitter_routes_code_by_extension_and_infers_language():
    splitter = AutoSplitter(chunk_size=80, chunk_overlap=0)
    splitter.init_components()
    document = Document(
        content=(
            "def foo():\n    return 1\n\n"
            "def bar():\n    return 2\n\n"
            "class Baz:\n    def method(self):\n        return 3\n"
        ),
        metadata={"file_path": "module.py"},
    )

    chunks = splitter.execute(splitter.input_schema(documents=[document]))["documents"]

    assert len(chunks) >= 2
    assert any("def foo" in chunk.content for chunk in chunks)
    assert any(chunk.metadata["splitter_strategy"] == "code" for chunk in chunks)


def test_auto_splitter_falls_back_to_recursive_for_plain_text():
    splitter = AutoSplitter(chunk_size=25, chunk_overlap=0)
    splitter.init_components()
    document = Document(
        content="alpha beta gamma delta epsilon zeta eta theta",
        metadata={"file_path": "notes.txt"},
    )

    chunks = splitter.execute(splitter.input_schema(documents=[document]))["documents"]

    assert len(chunks) > 1
    assert all(chunk.metadata["splitter_strategy"] == "recursive_character" for chunk in chunks)
    assert all("chunk_index" in chunk.metadata for chunk in chunks)


def test_auto_splitter_explicit_metadata_strategy_wins():
    splitter = AutoSplitter(chunk_size=20, chunk_overlap=0)
    splitter.init_components()
    document = Document(
        content="# Not a header route\nalpha beta gamma delta epsilon",
        metadata={"file_path": "notes.md", "splitter_strategy": "recursive_character"},
    )

    chunks = splitter.execute(splitter.input_schema(documents=[document]))["documents"]

    assert len(chunks) > 1
    assert all(chunk.metadata["splitter_strategy"] == "recursive_character" for chunk in chunks)


def test_auto_splitter_combined_rule_requires_matching_metadata():
    splitter = AutoSplitter(
        chunk_size=20,
        chunk_overlap=0,
        infer_from_content=False,
        rules=[
            AutoSplitterRule(
                strategy=AutoSplitterStrategy.CODE,
                file_types=["python"],
                metadata={"category": "tests"},
            )
        ],
    )
    splitter.init_components()
    document = Document(
        content="alpha beta gamma delta epsilon zeta eta theta",
        metadata={"file_type": "python", "category": "docs"},
    )

    chunks = splitter.execute(splitter.input_schema(documents=[document]))["documents"]

    assert len(chunks) > 1
    assert all(chunk.metadata["splitter_strategy"] == "recursive_character" for chunk in chunks)


def test_auto_splitter_combined_rule_matches_when_metadata_matches():
    splitter = AutoSplitter(
        rules=[
            AutoSplitterRule(
                strategy=AutoSplitterStrategy.CODE,
                file_types=["python"],
                metadata={"category": "tests"},
            )
        ],
    )
    splitter.init_components()
    document = Document(
        content="def test_example():\n    assert True\n",
        metadata={"file_type": "python", "category": "tests"},
    )

    chunks = splitter.execute(splitter.input_schema(documents=[document]))["documents"]

    assert chunks
    assert all(chunk.metadata["splitter_strategy"] == "code" for chunk in chunks)


def test_auto_splitter_serialization_excludes_runtime_splitter():
    splitter = AutoSplitter(chunk_size=50, chunk_overlap=5)
    splitter.init_components()

    serialized = splitter.to_dict()

    assert "splitter" not in serialized
    assert serialized["chunk_size"] == 50
    assert serialized["rules"]


def test_auto_splitter_uses_component_default_rules():
    splitter = AutoSplitter()

    assert splitter.rules == _default_rules()


def test_auto_splitter_yaml_roundtrip(tmp_path):
    workflow = Workflow(flow=Flow(nodes=[AutoSplitter(id="auto_splitter", chunk_size=100, chunk_overlap=0)]))
    yaml_path = tmp_path / "auto_splitter.yaml"

    workflow.to_yaml_file(yaml_path)
    loaded = Workflow.from_yaml_file(str(yaml_path), init_components=True)

    loaded_node = loaded.flow.nodes[0]
    assert isinstance(loaded_node, AutoSplitter)
    assert loaded_node.chunk_size == 100
    assert loaded_node.splitter is not None


def test_text_converter_to_auto_splitter_workflow_pipeline(tmp_path):
    source = tmp_path / "guide.md"
    source.write_text("# Title\nIntro\n## Details\nBody", encoding="utf-8")

    converter = TextFileConverter(id="converter")
    splitter = AutoSplitter(
        id="auto_splitter",
        depends=[NodeDependency(converter)],
        input_mapping={"documents": converter.outputs.documents},
    )
    workflow = Workflow(flow=Flow(nodes=[converter, splitter]))

    result = workflow.run_sync({"file_paths": [str(source)]})

    assert result.status == RunnableStatus.SUCCESS
    split_output = result.output["auto_splitter"]["output"]
    documents = split_output["documents"]
    assert len(documents) == 2
    assert documents[0]["metadata"]["splitter_strategy"] == "markdown_header"
    assert documents[0]["metadata"]["file_path"] == str(source)
