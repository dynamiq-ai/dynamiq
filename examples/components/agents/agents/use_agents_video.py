# flake8: noqa
"""
Upload a local video to an Agent and ask a question about it.

Important: unlike `images`/`files` passed via `input_data`, video content is NOT
auto-detected and injected into the prompt by Agent.execute() today. The reliable way to get
video in front of the model is the same pattern used for any custom vision input: build a
`VisionMessage` with Jinja placeholders, pass it as `input_message` when constructing the
Agent, then fill the placeholders via `agent.run(input_data={...})`. This mirrors how
`tests/integration/nodes/agents/test_agent_input_message.py::test_custom_vision_agent_workflow`
wires a VisionMessage into an agent.

Native video input is provider-specific -- only Gemini (via litellm) accepts it today. The
LLM's `is_video_input_supported` flag is checked up front so this fails with a clear message
instead of silently sending a video a text-only model will ignore.

Usage:
    1. Set VIDEO_FILE below to a local video path (mp4, mov, webm, etc.).
    2. Set your Gemini API key: export GEMINI_API_KEY=...
    3. python examples/components/agents/agents/use_agents_video.py
"""
import sys

from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.agents.utils import bytes_to_data_url
from dynamiq.prompts import (
    MessageRole,
    VisionMessage,
    VisionMessageFileContent,
    VisionMessageFileData,
    VisionMessageTextContent,
)
from examples.llm_setup import setup_llm

# Point this at a local video file before running.
VIDEO_FILE = ""

# Reads from the GEMINI_API_KEY env var -- never hardcode a real key in this file.
# export GEMINI_API_KEY=...

MODEL_NAME = "gemini/gemini-3.5-flash"


def build_video_input_message() -> VisionMessage:
    """VisionMessage with Jinja placeholders, filled in later via agent.run(input_data=...)."""
    return VisionMessage(
        content=[
            VisionMessageTextContent(text="{{question}}"),
            VisionMessageFileContent(file=VisionMessageFileData(file_data="{{video_data_url}}")),
        ],
        role=MessageRole.USER,
    )


def run_agent_with_local_video(video_path: str, question: str) -> str:
    llm = setup_llm(model_provider="gemini", model_name=MODEL_NAME)

    if not llm.is_video_input_supported:
        raise RuntimeError(
            f"Model '{MODEL_NAME}' is not registered as supporting video input. "
            "Pick a Gemini model, or add it to model_registry.json with "
            '"supports_video_input": true if you know it supports video.'
        )

    agent = Agent(
        name="VideoQAAgent",
        llm=llm,
        input_message=build_video_input_message(),
    )

    with open(video_path, "rb") as f:
        video_data_url = bytes_to_data_url(f.read())

    wf = Workflow(flow=Flow(nodes=[agent]))
    result = wf.run(input_data={"question": question, "video_data_url": video_data_url})

    return result.output[agent.id]["output"]["content"]


def main():
    video_path = VIDEO_FILE or (sys.argv[1] if len(sys.argv) > 1 else "")
    if not video_path:
        raise SystemExit(
            "Set VIDEO_FILE at the top of this script, or pass a video path as the first argument."
        )

    answer = run_agent_with_local_video(video_path, question="What is happening in this video? Be specific.")
    print(answer)


if __name__ == "__main__":
    main()
