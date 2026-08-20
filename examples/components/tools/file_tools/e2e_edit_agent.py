"""End-to-end check of FileWriteTool's 'edit' action driven by a real agent.

Exercises the behaviour added on fix/file-edit-ambiguous-anchors:

1. Ambiguous anchor  - a find string that matches several places is rejected
                       instead of silently editing the first match, and the
                       agent is expected to recover (add context, or replace_all).
2. CRLF file         - the agent writes find strings with bare newlines; the tool
                       matches on an LF view and restores CRLF on write, so the
                       endings it never touched come back unchanged.
3. Line reporting    - the success summary names the lines that changed, in the
                       file as written.

Run with a provider key already in the environment, e.g.:

    OPENAI_API_KEY=... python examples/components/tools/file_tools/e2e_edit_agent.py
    python examples/components/tools/file_tools/e2e_edit_agent.py --provider claude
"""

import argparse
import sys

from dotenv import load_dotenv
from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.file import FileStoreConfig
from dynamiq.storages.file.in_memory import InMemoryFileStore
from examples.llm_setup import setup_llm

load_dotenv()  # provider keys (OPENAI_API_KEY / ANTHROPIC_API_KEY / ...) from a local .env

AGENT_ROLE = (
    "You are a careful code editor. You edit files in place with the file-write tool using its "
    "'edit' action. Never rewrite a whole file when an edit will do. If an edit is rejected, read "
    "the error, fix your find string, and retry - do not give up and do not fall back to "
    "'write' unless editing is impossible."
)

# Two identical `return None` bodies: a bare "    return None" is ambiguous on purpose.
CONFIG_PY = """import os


def load_timeout():
    value = os.environ.get("TIMEOUT")
    if value is None:
        return None
    return int(value)


def load_retries():
    value = os.environ.get("RETRIES")
    if value is None:
        return None
    return int(value)


DEBUG = False
"""

# CRLF file - the agent will naturally write "\n" in its find strings.
SETTINGS_INI_CRLF = "[server]\r\n" "host = localhost\r\n" "port = 8080\r\n" "\r\n" "[logging]\r\n" "level = INFO\r\n"

# Each case: 'check' inspects the final file; 'expect_messages' are substrings that must
# appear in some file-write result, so a case cannot pass without actually reaching the
# code path it is meant to exercise.
CASES = [
    {
        "name": "ambiguous-anchor",
        "path": "config.py",
        "seed": CONFIG_PY,
        "prompt": (
            "In the file 'config.py', change ONLY the `return None` inside `load_retries` so it "
            "returns 3 instead. Leave `load_timeout` exactly as it is. Start with the shortest "
            "find string that identifies the line you want - do not quote surrounding lines "
            "unless a first attempt fails. Use the file-write tool's edit action."
        ),
        "check": lambda text: (
            text.count("return None") == 1
            and "return 3" in text
            and text.split("def load_retries")[0].count("return None") == 1
        ),
        "expect_messages": ["ambiguous find string"],
        "expect": "ambiguity is reported, then load_retries returns 3 and load_timeout is untouched",
    },
    {
        "name": "replace-all",
        "path": "config.py",
        "seed": CONFIG_PY,
        "prompt": (
            "In the file 'config.py', every `return None` should become `return 0`. Change all of "
            "them in a single edit operation. Use the file-write tool's edit action."
        ),
        "check": lambda text: "return None" not in text and text.count("return 0") == 2,
        "expect_messages": ["2 replacement(s)"],
        "expect": "both `return None` become `return 0` in one replace_all edit",
    },
    {
        "name": "crlf-multiline",
        "path": "settings.ini",
        "seed": SETTINGS_INI_CRLF,
        "prompt": (
            "In the file 'settings.ini', replace the two lines `host = localhost` and "
            "`port = 8080` with the single line `bind = localhost:9090`. Do it with one edit "
            "whose find string spans both lines. Use the file-write tool's edit action."
        ),
        "check": lambda text: (
            "bind = localhost:9090" in text
            and "host = localhost" not in text
            and "port = 8080" not in text
            and "\n" not in text.replace("\r\n", "")  # every newline is still a CRLF
        ),
        "expect_messages": ["replacement(s)"],
        "expect": "a multi-line LF find string matches the CRLF file and CRLF endings survive",
    },
]


def build_agent(llm, backend: InMemoryFileStore) -> Agent:
    return Agent(
        name="FileEditAgent",
        id="file-edit-agent",
        llm=llm,
        role=AGENT_ROLE,
        file_store=FileStoreConfig(enabled=True, backend=backend, agent_file_write_enabled=True),
        inference_mode=InferenceMode.FUNCTION_CALLING,
        max_loops=8,
    )


def edit_tool_calls(tracing: TracingCallbackHandler) -> list[dict]:
    """Every file-write invocation the agent made, as (input, output) pairs."""
    calls = []
    for run in tracing.runs.values():
        name = (run.metadata or {}).get("node", {}).get("name", "")
        if "file-write" in name.lower() or "filewrite" in name.lower().replace("-", ""):
            calls.append({"input": run.input, "output": run.output, "status": run.status})
    return calls


def run_case(case: dict, provider: str, model: str) -> bool:
    print(f"\n{'=' * 78}\nCASE: {case['name']}  ->  expected: {case['expect']}\n{'=' * 78}")

    backend = InMemoryFileStore()
    backend.store(case["path"], case["seed"].encode("utf-8"), content_type="text/plain", overwrite=True)

    llm = setup_llm(model_provider=provider, model_name=model, temperature=0)
    agent = build_agent(llm, backend)
    workflow = Workflow(flow=Flow(nodes=[agent]))

    tracing = TracingCallbackHandler()
    result = workflow.run(
        input_data={"input": case["prompt"]},
        config=RunnableConfig(callbacks=[tracing]),
    )

    print("--- agent answer ---")
    print(result.output[agent.id]["output"]["content"])

    print("\n--- file-write tool calls ---")
    calls = edit_tool_calls(tracing)
    if not calls:
        print("  (none captured in tracing - the agent may have used a different tool)")
    for i, call in enumerate(calls, 1):
        print(f"  [{i}] status={call['status']}")
        print(f"      input : {call['input']}")
        print(f"      output: {call['output']}")

    final = backend.retrieve(case["path"]).decode("utf-8")
    print("\n--- resulting file (repr, so line endings are visible) ---")
    for line in final.splitlines(keepends=True):
        print(f"  {line!r}")

    transcript = " ".join(str((c["output"] or {}).get("content", "")) for c in calls)
    unseen = [m for m in case.get("expect_messages", []) if m not in transcript]

    file_ok = case["check"](final)
    ok = file_ok and not unseen
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'} - {case['expect']}")
    if not file_ok:
        print("  file content does not match the expectation")
    if unseen:
        # The file may still be correct - this means the agent never provoked the
        # behaviour under test, so the run proves nothing about it.
        print(f"  code path not exercised: no tool result contained {unseen}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="gpt", help="claude | gpt | gemini | groq | cohere")
    parser.add_argument("--model", default=None, help="model name; defaults per provider")
    parser.add_argument("--case", default=None, help="run only this case by name")
    args = parser.parse_args()

    defaults = {
        "gpt": "gpt-4o",
        "claude": "claude-sonnet-5",
        "gemini": "gemini/gemini-1.5-pro-latest",
        "groq": "groq/llama3-70b-8192",
        "cohere": "command-r-plus",
    }
    model = args.model or defaults.get(args.provider, "gpt-4o")

    cases = [c for c in CASES if args.case is None or c["name"] == args.case]
    if not cases:
        print(f"No case named {args.case!r}. Available: {[c['name'] for c in CASES]}")
        return 2

    results = {c["name"]: run_case(c, args.provider, model) for c in cases}

    print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
