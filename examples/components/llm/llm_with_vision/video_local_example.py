# flake8: noqa
"""
Send a local video file to an LLM node directly (no agent involved) and ask a question about it.

Native video input is provider-specific -- as of writing, only Gemini (via litellm) accepts
video content. This example uses `llm.is_video_input_supported` to fail fast with a clear
message if you point it at a model that can't actually see the video, instead of silently
sending bytes the model will ignore.

Usage:
    1. Set VIDEO_FILE below to a local video path (mp4, mov, webm, etc.).
    2. Set your Gemini API key: export GEMINI_API_KEY=...
    3. python examples/components/llm/llm_with_vision/video_local_example.py
"""
import mimetypes
import sys
from base64 import b64encode

from dynamiq.connections import Gemini as GeminiConnection
from dynamiq.nodes.llms import Gemini
from dynamiq.prompts import (
    Prompt,
    VisionMessage,
    VisionMessageFileContent,
    VisionMessageFileData,
    VisionMessageTextContent,
)

# Point this at a local video file before running.
VIDEO_FILE = ""

MODEL_NAME = "gemini/gemini-3.5-flash"


def encode_video(video_path: str) -> str:
    """Read a local video file and return it as a base64 data URL."""
    mime_type, _ = mimetypes.guess_type(video_path)
    if not mime_type or not mime_type.startswith("video/"):
        raise ValueError(f"Could not determine a video mime type for '{video_path}' (got: {mime_type}).")

    with open(video_path, "rb") as f:
        encoded = b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


def get_prompt() -> Prompt:
    text_message = VisionMessageTextContent(text="{{question}}")
    video_message = VisionMessageFileContent(
        file=VisionMessageFileData(
            file_data="{{video_data_url}}",
            # Optional: sample fewer frames / trim the clip to control token cost.
            # video_metadata={"fps": 1, "start_offset": "0s", "end_offset": "30s"},
        )
    )
    vision_message = VisionMessage(content=[text_message, video_message], role="user")
    return Prompt(id="video-question", messages=[vision_message])


def run_gemini_local_video(video_path: str, question: str) -> str:
    llm = Gemini(
        name="Gemini Video Answer Generation",
        model=MODEL_NAME,
        prompt=get_prompt(),
        connection=GeminiConnection(),
    )

    if not llm.is_video_input_supported:
        raise RuntimeError(
            f"Model '{MODEL_NAME}' is not registered as supporting video input. "
            "Pick a Gemini model, or add it to model_registry.json with "
            '"supports_video_input": true if you know it supports video.'
        )

    video_data_url = encode_video(video_path)

    output = llm.execute(
        input_data={
            "question": question,
            "video_data_url": video_data_url,
        }
    )
    return output.get("content") or output


def main():
    video_path = VIDEO_FILE or (sys.argv[1] if len(sys.argv) > 1 else "")
    if not video_path:
        raise SystemExit(
            "Set VIDEO_FILE at the top of this script, or pass a video path as the first argument."
        )

    answer = run_gemini_local_video(video_path, question="What is happening in this video? Be specific.")
    print(answer)


if __name__ == "__main__":
    main()
