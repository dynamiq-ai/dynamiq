"""Presentation-creation agent — exercises the tool-calling reliability fixes.

Builds a multi-slide pptxgenjs Node.js script via an Agent in FUNCTION_CALLING
mode, switching between Anthropic and OpenAI through a single ``--provider`` flag.

This is the same shape of task that previously triggered the "model emits only
{thought: ...} and stops" failure on Anthropic. With the changes in
schema_generator.py + base.py + anthropic.py + openai.py + json_parser.py +
agent.py, it should now complete reliably under both providers.

Usage:
    # Anthropic (Claude)
    ANTHROPIC_API_KEY=sk-ant-... python -m examples.components.agents.presentation_agent --provider anthropic

    # OpenAI (GPT)
    OPENAI_API_KEY=sk-... python -m examples.components.agents.presentation_agent --provider openai

    # Switch the model explicitly
    python -m examples.components.agents.presentation_agent --provider anthropic --model anthropic/claude-opus-4-7
    python -m examples.components.agents.presentation_agent --provider openai --model openai/gpt-4.1

    # Or change the topic
    python -m examples.components.agents.presentation_agent --provider anthropic \
        --topic "intro to Formula 1 for kids, 4 slides, racing colors"

Output:
    The agent writes a Node.js pptxgenjs script to ./outputs/deck.js. Run it
    afterwards with `cd outputs && npm install pptxgenjs && node deck.js` to
    produce the .pptx file.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dynamiq.nodes.agents.agent import Agent
from dynamiq.nodes.tools.file_tools import FileWriteTool
from dynamiq.nodes.types import InferenceMode

DEFAULT_TOPIC = (
    "A fun, kid-friendly presentation explaining Formula 1 racing. "
    "Exactly 3 slides: (1) title slide with a bold red/yellow racing theme, "
    "(2) 'How an F1 car works' with 3-4 short bullet points, "
    "(3) 'Famous F1 drivers' with 2-3 names and one-line bios. "
    "Use the pptxgenjs Node.js library. Make colors vibrant and child-friendly."
)

DEFAULT_INSTRUCTION = (
    "Write a complete, self-contained Node.js script using the pptxgenjs library "
    "that produces a presentation matching the user's topic. "
    "The script must `require('pptxgenjs')`, build the slides programmatically, "
    "and call `pres.writeFile({{ fileName: 'deck.pptx' }})` at the end. "
    "Save the script to outputs/deck.js using the FileWrite tool. "
    "Topic:\n{topic}"
)


def build_llm(provider: str, model: str | None):
    """Construct an Anthropic or OpenAI LLM with sensible defaults.

    The new tool-calling reliability hooks live on these provider classes:
    - Anthropic: forced tool_choice, schema cleaning, strict_allowlist support.
    - OpenAI: nullable-required conversion, strict-by-default on tool schemas.
    """
    provider = provider.lower()
    if provider == "anthropic":
        from dynamiq.nodes.llms.anthropic import Anthropic

        return Anthropic(
            model=model or "anthropic/claude-opus-4-7",
            temperature=0.3,
            max_tokens=8000,
            # Demonstrates the new fields; defaults are conservative.
            force_tool_choice=True,
            strict_allowlist={"FileWrite"},
        )

    if provider == "openai":
        from dynamiq.nodes.llms.openai import OpenAI

        return OpenAI(
            model=model or "openai/gpt-4.1",
            temperature=0.3,
            max_tokens=8000,
        )

    raise SystemExit(f"Unknown provider: {provider!r}. Use 'anthropic' or 'openai'.")


def check_api_key(provider: str) -> None:
    var = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}[provider.lower()]
    if not os.environ.get(var):
        sys.exit(f"{var} is not set. Export your {provider.title()} API key and rerun:\n" f"    export {var}=...\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--provider",
        choices=("anthropic", "openai"),
        default=os.environ.get("PRESENTATION_PROVIDER", "anthropic"),
        help="Which LLM provider to use (default: anthropic, or $PRESENTATION_PROVIDER).",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Override model id (default: anthropic/claude-opus-4-7 or openai/gpt-4.1).",
    )
    p.add_argument(
        "--topic",
        default=DEFAULT_TOPIC,
        help="Presentation topic + content brief.",
    )
    p.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory the FileWrite tool writes into (default: ./outputs).",
    )
    p.add_argument(
        "--max-loops",
        type=int,
        default=8,
        help="Max agent loop iterations (default: 8).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    check_api_key(args.provider)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    file_write = FileWriteTool(
        name="FileWrite",
        description=(
            "Write text content to a file. Use this to save the pptxgenjs script. "
            "The file_path is relative to the current working directory."
        ),
    )

    llm = build_llm(args.provider, args.model)

    agent = Agent(
        name="PresentationBuilder",
        llm=llm,
        tools=[file_write],
        role=(
            "You are an expert presentation engineer. You write polished "
            "pptxgenjs Node.js scripts that produce clean, on-brand slide decks. "
            "Be decisive: one tool call to write the script, then finish."
        ),
        inference_mode=InferenceMode.FUNCTION_CALLING,
        max_loops=args.max_loops,
    )

    instruction = DEFAULT_INSTRUCTION.format(topic=args.topic.strip())

    print(f"[provider={args.provider}] [model={llm.model}] starting agent…\n")
    print("Instruction:")
    print(instruction)
    print()

    # Hand the relative output path through so the agent doesn't invent one.
    relative_script = str(Path(args.output_dir) / "deck.js")
    result = agent.run(
        input_data={"input": (f"{instruction}\n\n" f"Save the script to exactly this path: {relative_script}")},
        config=None,
    )

    print("\n=== Agent response ===")
    print(result.output)

    written = output_dir / "deck.js"
    if written.exists():
        size = written.stat().st_size
        print(f"\nWrote {written} ({size:,} bytes).")
        print("Run it with:")
        print(f"  cd {output_dir} && npm install pptxgenjs && node deck.js")
    else:
        print(
            f"\nNo file at {written}. The agent may have written elsewhere; "
            "check the agent response for the actual path it used."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
