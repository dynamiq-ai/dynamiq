from dynamiq.components.audio.stt.normalize import (
    assign_word_speakers,
    build_transcript,
    render_transcript,
    segments_from_words,
    speaker_label,
)
from dynamiq.types.audio import TranscriptSegment, TranscriptWord


def word(text, start, end, speaker=None, kind="word"):
    return TranscriptWord(word=text, start=start, end=end, speaker=speaker, type=kind)


def test_speaker_label_strips_provider_prefixes():
    assert speaker_label("0") == "Speaker 0"
    assert speaker_label("A") == "Speaker A"
    assert speaker_label("speaker_1") == "Speaker 1"
    assert speaker_label("spk_2") == "Speaker 2"
    assert speaker_label("S3") == "Speaker 3"
    assert speaker_label("Alice") == "Speaker Alice"


def test_segments_from_words_groups_by_speaker_and_attaches_punctuation():
    words = [
        word("Hi", 0.0, 0.2, "0"),
        word(",", 0.2, 0.2, "0", kind="punctuation"),
        word("there", 0.3, 0.5, "0"),
        word("Hello", 1.0, 1.3, "1"),
    ]

    segments = segments_from_words(words)

    assert [segment.text for segment in segments] == ["Hi, there", "Hello"]
    assert [segment.speaker for segment in segments] == ["0", "1"]
    assert (segments[0].start, segments[0].end) == (0.0, 0.5)
    assert all(segment.synthetic for segment in segments)


def test_assign_word_speakers_copies_segment_speakers_by_time():
    segments = [
        TranscriptSegment(id="0", text="Hi", start=0.0, end=1.0, speaker="A"),
        TranscriptSegment(id="1", text="Yo", start=1.0, end=2.0, speaker="B"),
    ]
    words = [word("Hi", 0.1, 0.3), word("Yo", 1.5, 1.7), word("late", 5.0, 5.2)]

    assign_word_speakers(words, segments)

    assert [item.speaker for item in words] == ["A", "B", None]


def test_assign_word_speakers_keeps_provider_word_speakers():
    segments = [TranscriptSegment(id="0", text="Hi", start=0.0, end=1.0, speaker="A")]
    words = [word("Hi", 0.1, 0.3, "Z")]

    assign_word_speakers(words, segments)

    assert words[0].speaker == "Z"


def test_render_transcript_merges_consecutive_segments_of_one_speaker():
    segments = [
        TranscriptSegment(id="0", text="Hi.", speaker="0"),
        TranscriptSegment(id="1", text="Again.", speaker="0"),
        TranscriptSegment(id="2", text="Yes.", speaker="1"),
        TranscriptSegment(id="3", text="   ", speaker="1"),
    ]

    assert render_transcript(segments, "ignored") == "Speaker 0: Hi. Again.\nSpeaker 1: Yes."


def test_render_transcript_without_speakers_returns_content():
    segments = [TranscriptSegment(id="0", text="Hi.")]

    assert render_transcript(segments, "Hi.") == "Hi."


def test_build_transcript_without_segments_creates_a_single_synthetic_one():
    transcript = build_transcript(content="Hello world", duration=2.0, language="en")

    assert transcript.transcript == "Hello world"
    assert transcript.languages == ["en"]
    assert transcript.speakers == []
    assert len(transcript.segments) == 1
    assert transcript.segments[0].synthetic is True
    assert transcript.segments[0].end == 2.0


def test_build_transcript_derives_segments_speakers_and_duration_from_words():
    words = [word("Hi", 0.0, 0.4, "speaker_0"), word("Bye", 1.0, 1.4, "speaker_1")]

    transcript = build_transcript(content="Hi Bye", words=words, languages=["en", "fr"])

    assert [segment.speaker for segment in transcript.segments] == ["speaker_0", "speaker_1"]
    assert [speaker.label for speaker in transcript.speakers] == ["Speaker 0", "Speaker 1"]
    assert transcript.transcript == "Speaker 0: Hi\nSpeaker 1: Bye"
    assert transcript.duration == 1.4
    assert transcript.language == "en"
