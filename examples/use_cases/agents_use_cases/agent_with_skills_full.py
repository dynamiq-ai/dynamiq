import html
import io
import os
from datetime import datetime

from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections import Dynamiq
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig
from dynamiq.skills import SkillsBackendType
from dynamiq.utils.logger import logger

DYNAMIQ_SANDBOX_URL = os.environ.get("DYNAMIQ_URL", "https://api.sandbox.getdynamiq.ai")
HUMANIZER_SKILL_ID = "cfb2ddd9-7b5b-457f-9ef6-666db9c45eb3"
HUMANIZER_VERSION_ID = "adcf8695-31b0-4a50-ba68-3606b5158e7c"

AGENT_ROLE = """
You have access to skills via the SkillsTool. Use only:
- action="list" to see available skills
- action="get" and skill_name="..." to load full skill content

After get, apply the skill's guidelines in your reasoning
 and provide the result in your final answer.
 Do not call the tool again with content to transform.
Format responses in Markdown.
"""


def create_agent(connection: Dynamiq, tracing_handler=None) -> Agent:
    """Create agent with Dynamiq API skills (list + get)."""
    llm = OpenAI(model="gpt-4o", temperature=0.7, max_tokens=4096)
    agent = Agent(
        name="SkillsAgent",
        llm=llm,
        tools=[],
        role=AGENT_ROLE,
        max_loops=10,
        inference_mode=InferenceMode.XML,
        skills={
            "enabled": True,
            "backend": {
                "type": SkillsBackendType.Dynamiq,
                "connection": connection,
            },
            "whitelist": [
                {
                    "id": HUMANIZER_SKILL_ID,
                    "name": "humanizer",
                    "description": "Remove signs of AI-generated writing from text.",
                    "version_id": HUMANIZER_VERSION_ID,
                },
            ],
        },
    )
    logger.info("Agent created with Dynamiq skills (list, get)")
    return agent


def run_workflow(
    agent: Agent,
    prompt: str,
    files: list[io.BytesIO] | None = None,
    tracing_handler=None,
) -> tuple[str, object]:
    """Run agent once."""
    input_data = {"input": prompt}
    if files:
        input_data["files"] = files
    callbacks = [tracing_handler] if tracing_handler else None
    run_config = RunnableConfig(callbacks=callbacks) if callbacks else None
    result = agent.run(input_data=input_data, config=run_config)
    return result.output.get("content", ""), result.output.get("files")


def main():
    print("\n" + "=" * 80)
    print("Skills example: list, get (Dynamiq API)")
    print("=" * 80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    api_key = os.environ.get("DYNAMIQ_API_KEY")
    if not api_key:
        print("Set DYNAMIQ_API_KEY to run this example.")
        return
    connection = Dynamiq(base_url=DYNAMIQ_SANDBOX_URL, api_key=api_key)
    tracing_handler = TracingCallbackHandler()
    agent = create_agent(connection, tracing_handler)

    prompt = (
        "1) List available skills. "
        "2) Get the full content of the humanizer skill. "
        "3) Summarize what the skill is for and how to apply it."
    )
    print("Prompt:", prompt, "\n")

    output, files = run_workflow(agent=agent, prompt=prompt, tracing_handler=tracing_handler)

    print("=" * 80)
    print("Agent output")
    print("=" * 80)
    print(html.unescape(output))
    print("=" * 80)
    if files:
        n = len(files) if isinstance(files, (list, dict)) else 1
        print(f"Result also included {n} file(s).")
    print(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    main()
