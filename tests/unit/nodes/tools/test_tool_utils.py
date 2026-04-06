import pytest

from dynamiq.nodes.tools.utils import sanitize_filename


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("file.txt", "file.txt"),
        ("document.pdf", "document.pdf"),
        ("image_2024.png", "image_2024.png"),
        ("my document.txt", "my document.txt"),
        ("file-name_v2 (copy).txt", "file-name_v2 (copy).txt"),
    ],
)
def test_sanitize_filename_normal_filenames(filename, expected):
    """Normal filenames should pass through unchanged."""
    assert sanitize_filename(filename) == expected


@pytest.mark.parametrize(
    "filename,default,expected",
    [
        ("", None, ""),
        (None, None, ""),
        ("", "fallback", "fallback"),
        (None, "unnamed", "unnamed"),
        ("...", None, ""),
        ("...", "fallback", "fallback"),
        ("   ", None, ""),
        ("   ", "fallback", "fallback"),
    ],
)
def test_sanitize_filename_empty_or_invalid_with_default(filename, default, expected):
    """Empty or invalid filenames should return empty or default value."""
    assert sanitize_filename(filename, default=default) == expected


@pytest.mark.parametrize(
    "path,expected",
    [
        ("../../../etc/passwd", "passwd"),
        ("subdir/../../../malicious.sh", "malicious.sh"),
        ("/etc/passwd", "passwd"),
        ("/home/user/documents/file.txt", "file.txt"),
        ("dir/file.txt", "file.txt"),
    ],
)
def test_sanitize_filename_path_traversal_extracts_filename(path, expected):
    """Path traversal and absolute paths should have only filename extracted."""
    assert sanitize_filename(path) == expected


@pytest.mark.parametrize(
    "path",
    [
        "..\\..\\..\\Windows\\System32\\config",
        "..\\..\\Windows\\System32\\evil.dll",
    ],
)
def test_sanitize_filename_windows_path_is_sanitized(path):
    """Windows-style paths should be sanitized (backslashes replaced)."""
    result = sanitize_filename(path)
    assert "/" not in result
    assert "\\" not in result


@pytest.mark.parametrize(
    "filename,expected",
    [
        (".hidden", "hidden"),
        ("..hidden", "hidden"),
        ("...secret", "secret"),
    ],
)
def test_sanitize_filename_leading_dots_are_removed(filename, expected):
    """Leading dots (hidden files) should be removed."""
    assert sanitize_filename(filename) == expected


def test_sanitize_filename_null_bytes_are_removed():
    """Null bytes should be removed from filenames."""
    assert "\x00" not in sanitize_filename("file\x00.txt")


def test_sanitize_filename_control_characters_are_removed():
    """Control characters should be removed from filenames."""
    result = sanitize_filename("file\x01\x02\x03.txt")
    assert "\x01" not in result
    assert "\x02" not in result
    assert "\x03" not in result


@pytest.mark.parametrize(
    "name,expected",
    [
        ("../../../etc/cron.d/malicious", "malicious"),
        ("subdir/../../outside.txt", "outside.txt"),
    ],
)
def test_sanitize_prevents_zip_slip(name, expected):
    """sanitize_filename should prevent Zip Slip by extracting only the filename."""
    safe_name = sanitize_filename(name)
    assert "/" not in safe_name
    assert safe_name == expected


def test_sanitize_filename_unicode_filenames_are_preserved():
    """Unicode characters in filenames should be preserved."""
    result = sanitize_filename("документ.pdf")
    assert "документ" in result


def test_sanitize_filename_very_long_filename_is_handled():
    """Very long filenames should not cause errors."""
    long_name = "a" * 500 + ".txt"
    result = sanitize_filename(long_name)
    assert result is not None
    assert len(result) > 0
