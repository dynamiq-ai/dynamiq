---
name: hello-world
description: Conventions for greeting messages and welcome text. Use when writing user-facing welcome messages, onboarding copy, or greeting notifications.
---

# Hello World Conventions

When writing greeting messages for this project, follow these patterns.

## Welcome messages

Use a friendly, concise tone:

- "Welcome to [Product]! Let's get started."
- "Hey [Name], good to see you!"

Avoid:

- Overly formal greetings ("Dear valued customer...")
- Exclamation overload ("Welcome!!! So excited!!!")

## Onboarding copy

First-time user messages should:

1. State what the product does in one sentence
2. Offer a single clear next action
3. Provide a way to get help

Example:

"[Product] helps you [value prop]. Click 'New Project' to begin, or visit our docs if you need help."

## Optional script

To run the example script: use SkillsTool with action `run_script`, skill_name `hello-world`, script_path `scripts/run.py`, and optional arguments.

## File handling (run_script)

When a skill script reads or writes files, the agent manages paths as follows:

- **Paths**: All paths use forward slashes. FileStore paths (e.g. `data/input.html`, `.skills/hello-world/scripts/run.py`) are used when referring to files in the backend.
- **Input files**: To give the script access to files that are in the agent's FileStore, use `input_files`: a map from **FileStore path** → **sandbox path** (path the script will see). Example: `{"data/report.html": "input/report.html"}`. The backend copies each FileStore file into the sandbox at the given path before running the script.
- **Output files**: If the script writes files, list the **sandbox paths** to collect in `output_paths` (e.g. `["output/result.pptx"]`). After the run, those files are read from the sandbox and returned to the agent. Use `output_prefix` to set the FileStore prefix for stored paths (e.g. `output_prefix="generated/"` so the file is stored as `generated/result.pptx`).
- **Backend**: In YAML/DAG workflows, the runner typically uploads `.skills/` into the agent's FileStore; any other files (e.g. user uploads or pipeline outputs) can be stored under paths like `data/...`. The agent then passes those paths in `input_files` when calling `run_script` and uses the returned `output_files` (or stores them) as needed.
