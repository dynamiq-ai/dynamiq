from typing import Any

from dynamiq.types.audio import Speaker, Transcript, TranscriptSegment, TranscriptWord

_SPEAKER_PREFIXES = ("speaker_", "speaker ", "spk_")


def speaker_label(speaker_id: str) -> str:
    """Human label for a provider speaker id (``0``, ``A``, ``speaker_1``, ``spk_2``, ``S1``)."""
    raw = speaker_id.strip()
    lowered = raw.lower()
    for prefix in _SPEAKER_PREFIXES:
        if lowered.startswith(prefix):
            raw = raw[len(prefix) :]
            break
    else:
        if len(raw) > 1 and raw[0] in "Ss" and raw[1:].isdigit():
            raw = raw[1:]
    return f"Speaker {raw}"


def join_words(words: list[TranscriptWord]) -> str:
    """Join word tokens into text, attaching punctuation tokens to the preceding word."""
    parts: list[str] = []
    for word in words:
        if word.type == "punctuation" and parts:
            parts[-1] += word.word
        else:
            parts.append(word.word)
    return " ".join(parts).strip()


def segments_from_words(words: list[TranscriptWord]) -> list[TranscriptSegment]:
    """Rebuild segments from word-level output by grouping consecutive words of one speaker."""
    groups: list[list[TranscriptWord]] = []
    for word in words:
        if groups and groups[-1][0].speaker == word.speaker:
            groups[-1].append(word)
        else:
            groups.append([word])
    return [
        TranscriptSegment(
            id=str(index),
            text=join_words(group),
            start=group[0].start,
            end=group[-1].end,
            speaker=group[0].speaker,
            synthetic=True,
        )
        for index, group in enumerate(groups)
    ]


def assign_word_speakers(words: list[TranscriptWord], segments: list[TranscriptSegment]) -> None:
    """Copy segment speakers onto words when the provider diarizes at segment level only."""
    if not words or any(word.speaker is not None for word in words):
        return
    timed = [s for s in segments if s.speaker is not None and s.start is not None and s.end is not None]
    for word in words:
        if word.start is None:
            continue
        for segment in timed:
            if segment.start <= word.start <= segment.end:
                word.speaker = segment.speaker
                break


def collect_speakers(segments: list[TranscriptSegment], words: list[TranscriptWord]) -> list[Speaker]:
    """Distinct speakers in order of first appearance."""
    seen: dict[str, None] = {}
    for item in [*segments, *words]:
        if item.speaker is not None and item.speaker not in seen:
            seen[item.speaker] = None
    return [Speaker(id=speaker_id, label=speaker_label(speaker_id)) for speaker_id in seen]


def render_transcript(segments: list[TranscriptSegment], content: str) -> str:
    """Speaker-labelled text, merging consecutive segments of the same speaker into one line."""
    if not any(segment.speaker is not None for segment in segments):
        return content
    lines: list[tuple[str | None, list[str]]] = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        if lines and lines[-1][0] == segment.speaker:
            lines[-1][1].append(text)
        else:
            lines.append((segment.speaker, [text]))
    return "\n".join(
        f"{speaker_label(speaker) if speaker is not None else 'Unknown'}: {' '.join(texts)}" for speaker, texts in lines
    )


def build_transcript(
    *,
    content: str,
    segments: list[TranscriptSegment] | None = None,
    words: list[TranscriptWord] | None = None,
    language: str | None = None,
    languages: list[str] | None = None,
    duration: float | None = None,
    usage: dict[str, Any] | None = None,
    raw: dict[str, Any] | None = None,
) -> Transcript:
    """Assemble a ``Transcript`` from whatever a provider returned, filling derived fields.

    Segments are synthesized from words when the provider has none; a single segment is created
    when it returned only text. Duration falls back to the last known timestamp.
    """
    segments = list(segments or [])
    words = list(words or [])
    if not segments and words:
        segments = segments_from_words(words)
    if not segments and content.strip():
        segments = [TranscriptSegment(id="0", text=content.strip(), start=None, end=duration, synthetic=True)]
    assign_word_speakers(words, segments)

    if duration is None:
        ends = [item.end for item in [*segments, *words] if item.end is not None]
        duration = max(ends) if ends else None

    languages = list(languages or [])
    if language and language not in languages:
        languages.insert(0, language)
    if not language and languages:
        language = languages[0]

    return Transcript(
        content=content,
        transcript=render_transcript(segments, content),
        language=language,
        languages=languages,
        duration=duration,
        speakers=collect_speakers(segments, words),
        segments=segments,
        words=words,
        usage=dict(usage or {}),
        raw=dict(raw or {}),
    )


def optional_str(value: Any) -> str | None:
    """Stringify provider speaker ids (ints, letters, prefixed strings) while keeping None."""
    return None if value is None else str(value)
