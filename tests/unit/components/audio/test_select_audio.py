import io

import pytest

from dynamiq.components.audio.utils import select_audio_file


def named(name: str, content_type: str | None = None) -> io.BytesIO:
    file = io.BytesIO(b"\x00\x01")
    file.name = name
    if content_type:
        file.content_type = content_type
    return file


def test_single_file_is_used_as_is():
    file = named("recording.wav", "audio/wav")

    assert select_audio_file(file) is file


def test_bytes_are_used_as_is():
    payload = b"\x00\x01"

    assert select_audio_file(payload) is payload


def test_audio_wins_over_other_uploads_whatever_the_order():
    document = named("contract.pdf", "application/pdf")
    recording = named("meeting.wav", "audio/wav")

    assert select_audio_file([document, recording]) is recording


def test_extension_decides_when_the_store_has_no_content_type():
    document = named("contract.pdf")
    recording = named("meeting.m4a")

    assert select_audio_file([document, recording]) is recording


def test_video_counts_as_audio_because_providers_accept_it():
    document = named("slides.pptx", "application/vnd.ms-powerpoint")
    recording = named("standup.mp4", "video/mp4")

    assert select_audio_file([document, recording]) is recording


def test_first_audio_file_wins_when_several_are_uploaded():
    first = named("part-1.mp3", "audio/mpeg")
    second = named("part-2.mp3", "audio/mpeg")

    assert select_audio_file([first, second]) is first


def test_nothing_that_looks_like_audio_is_rejected_by_name():
    with pytest.raises(ValueError, match="none of them look like audio"):
        select_audio_file([named("contract.pdf", "application/pdf"), named("notes.txt", "text/plain")])


def test_unnamed_bytes_among_files_are_accepted_as_a_last_resort():
    # Sandboxes and ad-hoc callers hand over raw bytes with no name to judge; one of those is
    # still more likely to be the audio than a file that is provably a PDF.
    payload = b"\x00\x01"

    assert select_audio_file([named("contract.pdf", "application/pdf"), payload]) is payload


def test_empty_selection_is_none():
    assert select_audio_file([]) is None
    assert select_audio_file(None) is None


def test_the_store_stamping_octet_stream_does_not_hide_the_audio():
    """The agent's upload path stores files as application/octet-stream whenever the caller's
    BytesIO had no content type, which is the ordinary case. The name still says what it is."""
    recording = named("meeting.mp3", "application/octet-stream")

    assert select_audio_file([recording]) is recording


def test_a_generic_content_type_still_loses_to_a_named_recording():
    document = named("contract.pdf", "application/octet-stream")
    recording = named("meeting.wav", "application/octet-stream")

    assert select_audio_file([document, recording]) is recording


def test_a_generic_content_type_with_no_usable_name_is_a_last_resort():
    blob = named("blob", "application/octet-stream")

    assert select_audio_file([named("contract.pdf", "application/pdf"), blob]) is blob
