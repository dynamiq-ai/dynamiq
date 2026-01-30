import io
from datetime import datetime
from pathlib import Path

from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.tools import PythonCodeExecutor
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.file import FileStoreConfig, InMemoryFileStore
from dynamiq.utils.logger import logger

AGENT_ROLE = """
You have access to skills. Use the SkillsTool to:
- action="list" to see available skills
- action="get" and skill_name="..." to load full skill content when needed
- action="run_script" with skill_name and script_path to run a skill script in the sandbox

Format responses in Markdown.
"""

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SKILLS_DIR = PROJECT_ROOT / ".skills"


def upload_skills_to_filestore(file_store: InMemoryFileStore) -> None:
    """Upload skills from .skills/ into FileStore."""
    logger.info("Uploading skills to FileStore...")
    if not SKILLS_DIR.exists():
        logger.warning("No .skills directory at %s", SKILLS_DIR)
        return
    for skill_dir in SKILLS_DIR.iterdir():
        if not skill_dir.is_dir():
            continue
        for path in skill_dir.rglob("*"):
            if path.is_file():
                rel = path.relative_to(SKILLS_DIR)
                store_path = f".skills/{rel.as_posix()}"
                file_store.store(store_path, path.read_bytes())
                logger.info(" Uploaded %s", store_path)


def create_agent(file_store: InMemoryFileStore, tracing_handler=None) -> Agent:
    """Create agent with skills (default SkillsConfig: FileStore source + subprocess executor)."""
    python_tool = PythonCodeExecutor(name="python_executor", file_store=file_store)
    llm = OpenAI(model="gpt-4o", temperature=0.7, max_tokens=4096)
    file_store_config = FileStoreConfig(
        enabled=True,
        backend=file_store,
        agent_file_write_enabled=True,
    )
    agent = Agent(
        name="SkillsAgent",
        llm=llm,
        tools=[python_tool],
        role=AGENT_ROLE,
        max_loops=10,
        inference_mode=InferenceMode.XML,
        file_store=file_store_config,
        skills_enabled=True,
    )
    logger.info("Agent created with skills_enabled=True")
    return agent


def run_workflow(
    agent: Agent,
    prompt: str,
    files: list[io.BytesIO] = None,
    tracing_handler=None,
) -> tuple[str, object]:
    """Run agent once. Tool results are logged and passed to the agent as Observation."""
    input_data = {"input": prompt}
    if files:
        input_data["files"] = files
    callbacks = [tracing_handler] if tracing_handler else None
    run_config = RunnableConfig(callbacks=callbacks) if callbacks else None
    result = agent.run(input_data=input_data, config=run_config)
    return result.output.get("content", ""), result.output.get("files")


def main():
    print("\n" + "=" * 80)
    print("Skills example: list, get, run_script")
    print("=" * 80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    file_store = InMemoryFileStore()
    upload_skills_to_filestore(file_store)

    tracing_handler = TracingCallbackHandler()
    agent = create_agent(file_store, tracing_handler)

    prompt = (
        "1) List available skills. "
        "2) Get the full content of the hello-world skill. "
        "3) If the skill has a script, run it with run_script (skill_name=hello-world, script_path=scripts/run.py). "
        "Summarize what the skill is for and what the script output was."
    )
    print("Prompt:", prompt, "\n")

    output, files = run_workflow(agent=agent, prompt=prompt, tracing_handler=tracing_handler)

    print("=" * 80)
    print("Agent output")
    print("=" * 80)
    print(output)
    print("=" * 80)
    if files:
        n = len(files) if isinstance(files, (list, dict)) else 1
        print(f"Result also included {n} file(s).")
    print(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    main()
