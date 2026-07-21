from dynamiq.components.converters.utils import build_source_metadata


def test_build_source_metadata_prefers_explicit_source_without_mutating_input():
    original = {"file_path": "stored/item.md", "source": "e73554d0-d0e8-4b17-8512-a466f1e549a4"}

    result = build_source_metadata(original, "upload.md")

    assert result == original
    assert result is not original


def test_build_source_metadata_preserves_provider_url_without_creating_source():
    result = build_source_metadata(
        {
            "source": "",
            "dynamiq_item_source_provider_url": "https://source.example/article",
            "url": "https://fallback.example/article",
        },
        "upload.md",
    )

    assert result["file_path"] == "upload.md"
    assert "source" not in result
    assert result["dynamiq_item_source_provider_url"] == "https://source.example/article"
    assert result["url"] == "https://fallback.example/article"


def test_build_source_metadata_adds_only_file_path_for_missing_values():
    result = build_source_metadata({"file_path": None, "source": None}, "private-upload.txt")

    assert result["file_path"] == "private-upload.txt"
    assert "source" not in result


def test_build_source_metadata_does_not_add_source_when_absent():
    result = build_source_metadata({}, "private-upload.txt")

    assert result == {"file_path": "private-upload.txt"}
