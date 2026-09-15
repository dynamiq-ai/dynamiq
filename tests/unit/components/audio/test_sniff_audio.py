"""Identifying a recording by its contents.

An agent renames raw ``bytes`` uploads ``file_0.bin`` and the store stamps them
``application/octet-stream``, so neither the name nor the type says anything. The bytes do.
"""

import io

from dynamiq.components.audio.utils import prepare_audio_file, sniff_audio_extension

WAV = b"RIFF\x24\x00\x00\x00WAVEfmt "
MP3_ID3 = b"ID3\x04\x00\x00\x00\x00\x00\x00"
MP3_FRAME = b"\xff\xfb\x90\x64\x00\x00\x00\x00"
FLAC = b"fLaC\x00\x00\x00\x22"
OGG = b"OggS\x00\x02\x00\x00"
M4A = b"\x00\x00\x00\x20ftypM4A \x00\x00\x00\x00"
WEBM = b"\x1a\x45\xdf\xa3\x01\x00\x00\x00"


def test_every_container_a_provider_accepts_is_recognized():
    assert sniff_audio_extension(WAV) == "wav"
    assert sniff_audio_extension(MP3_ID3) == "mp3"
    assert sniff_audio_extension(MP3_FRAME) == "mp3"
    assert sniff_audio_extension(FLAC) == "flac"
    assert sniff_audio_extension(OGG) == "ogg"
    assert sniff_audio_extension(M4A) == "m4a"
    assert sniff_audio_extension(WEBM) == "webm"


def test_anything_else_is_not_guessed_at():
    assert sniff_audio_extension(b"%PDF-1.4 ...") is None
    assert sniff_audio_extension(b"") is None
    assert sniff_audio_extension(b"just some text") is None


def test_a_meaningless_extension_is_replaced_by_what_the_bytes_say():
    """OpenAI rejects `file_0.bin` with "Unsupported file format bin" — the name is the only
    thing it has to go on."""
    raw = io.BytesIO(MP3_ID3)
    raw.name = "file_0.bin"

    prepared = prepare_audio_file(raw, "audio.wav", "audio/wav")

    assert prepared.name == "file_0.mp3"


def test_a_name_the_provider_can_use_is_left_alone():
    raw = io.BytesIO(WAV)
    raw.name = "meeting.wav"

    assert prepare_audio_file(raw, "audio.wav", "audio/wav").name == "meeting.wav"


def test_an_unrecognized_payload_keeps_its_name():
    raw = io.BytesIO(b"%PDF-1.4")
    raw.name = "file_0.bin"

    assert prepare_audio_file(raw, "audio.wav", "audio/wav").name == "file_0.bin"


def test_bare_bytes_are_named_after_what_they_contain():
    assert prepare_audio_file(MP3_ID3, "audio.wav", "audio/wav").name == "audio.mp3"


def test_bare_unrecognized_bytes_fall_back_to_the_default_name():
    assert prepare_audio_file(b"%PDF-1.4", "audio.wav", "audio/wav").name == "audio.wav"
