from typing import TYPE_CHECKING

from dynamiq.utils.lazy_imports import lazy_exports

if TYPE_CHECKING:
    from .elevenlabs import ElevenLabsSTS, ElevenLabsTTS, Voices
    from .speech_to_text import SpeechToText
    from .text_to_speech import TextToSpeech
    from .whisper import WhisperSTT

_LAZY_EXPORTS = {
    "ElevenLabsSTS": ".elevenlabs",
    "ElevenLabsTTS": ".elevenlabs",
    "Voices": ".elevenlabs",
    "SpeechToText": ".speech_to_text",
    "TextToSpeech": ".text_to_speech",
    "WhisperSTT": ".whisper",
}

lazy_exports(__name__, _LAZY_EXPORTS, globals())

__all__ = [
    "ElevenLabsSTS",
    "ElevenLabsTTS",
    "Voices",
    "SpeechToText",
    "TextToSpeech",
    "WhisperSTT",
]
